import mnist_dataset
import tensorflow as tf
import os, json


def input_fn(mode, num_epochs, batch_size, input_context=None):
  
  datasets = mnist_dataset.load()

  _dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
            datasets['test'])

  if input_context:
    _dataset = _dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  _dataset = _dataset.shuffle(buffer_size= 2 * batch_size + 1).batch(batch_size).repeat(num_epochs)
  return _dataset

def model_fn(features, labels, mode, params):
  
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
  ])
  logits = model(features)

  predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
  score = tf.compat.v1.math.softmax(logits)
  predictions = {'logits': logits, 'classes': predicted_logit, 'probabilities': score}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  LEARNING_RATE = params['learning_rate']
  BATCH_SIZE = params['batch_size']
  num_workers = params['num_workers']

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)(labels, logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE/num_workers)

  # setting gradient_accumulation
  gradient_accumulation_multiplier = params['gradient_accumulation_multiplier']

  global_step = tf.train.get_global_step()

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  accum_grads = [tf.Variable(tf.zeros_like(t_var.initialized_value()), trainable=False, aggregation=tf.VariableAggregation.SUM) for t_var in tvars]

  def apply_accumulated_gradients(accum_grads, grads, tvars):
    accum_op= tf.group([accum_grad.assign_add(grad) for (accum_grad, grad) in zip(accum_grads, grads)])
    with tf.control_dependencies([accum_op]):
      normalized_accum_grads = [1.0*accum_grad/gradient_accumulation_multiplier for accum_grad in accum_grads]
      # global_step is not incremented inside optimizer.apply_gradients
      minimize_op= optimizer.apply_gradients(zip(normalized_accum_grads, tvars), global_step = None)
      with tf.control_dependencies([minimize_op]):
        zero_op= tf.group([accum_grad.assign(tf.zeros_like(accum_grad)) for accum_grad in accum_grads])
    
    return zero_op

  train_op = tf.cond(tf.math.equal(global_step % gradient_accumulation_multiplier, 0),
    lambda: apply_accumulated_gradients(accum_grads, grads, tvars),
    lambda: tf.group([accum_grad.assign_add(grad) for (accum_grad, grad) in zip(accum_grads, grads)])
  )

  # global_step is incremented here, regardless of the tf.cond branch
  train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])

  accuracy = tf.compat.v1.metrics.accuracy(labels, predicted_logit)
  eval_metric = { 'accuracy': accuracy }

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
      mode,
      loss=loss,
      train_op=None,
      eval_metric_ops=eval_metric,
      predictions=predictions
    )

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    # train_op=optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step()),
    train_op = train_op,
    eval_metric_ops=eval_metric,
    predictions=predictions
  )

if __name__ == "__main__":
  tfconfig = dict({
    'cluster': {
      'worker': ["192.168.1.10:2222", "192.168.1.11:2222"]
    },
    'task': {'type': 'worker', 'index': 0}
  })
  os.environ['TF_CONFIG'] = json.dumps(tfconfig)

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.RING)

  OUTDIR='tmp/multiworkergaccum'

  BATCH_SIZE = 50 ;#100
  NUM_EPOCHS = 5

  config = tf.estimator.RunConfig(
    train_distribute=strategy,
	eval_distribute=strategy,
    log_step_count_steps=100, 
    tf_random_seed=19830610,
    model_dir=OUTDIR
  )

  hparams = dict({'learning_rate': 1e-4, 'batch_size': BATCH_SIZE, 'gradient_accumulation_multiplier': 2, 'num_workers': len(tfconfig['cluster']['worker'])})

  classifier = tf.estimator.Estimator(
    model_fn=model_fn, config=config, params= hparams)

  train_spec = tf.estimator.TrainSpec(
    input_fn= lambda: input_fn(
      mode= tf.estimator.ModeKeys.TRAIN,
      num_epochs= NUM_EPOCHS,
      batch_size= BATCH_SIZE,
      input_context=tf.distribute.InputContext(len(tfconfig['cluster']['worker']), tfconfig['task']['index'])
    ),
    hooks= None,
  )

  eval_spec = tf.estimator.EvalSpec(
    input_fn= lambda: input_fn(
      mode= tf.estimator.ModeKeys.EVAL,
      num_epochs= 1,
      batch_size= 5000,
      input_context=tf.distribute.InputContext(len(tfconfig['cluster']['worker']), tfconfig['task']['index'])
    ),
    throttle_secs = 30,
    steps=None
  )

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  tf.estimator.train_and_evaluate(
    classifier,
    train_spec=train_spec,
    eval_spec=eval_spec,
  )

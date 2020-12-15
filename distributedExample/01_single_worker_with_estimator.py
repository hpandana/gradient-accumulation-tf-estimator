import mnist_dataset
import tensorflow as tf
import shutil


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

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)(labels, logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
  
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
    train_op=optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step()),
    eval_metric_ops=eval_metric,
    predictions=predictions
  )

if __name__ == "__main__":

  OUTDIR='tmp/singleworker'
  shutil.rmtree(OUTDIR, ignore_errors=True)

  BATCH_SIZE = 200
  NUM_EPOCHS = 5

  config = tf.estimator.RunConfig(
    log_step_count_steps=100,
    tf_random_seed=19830610,
    model_dir=OUTDIR
  )

  hparams = dict({'learning_rate': 1e-4, 'batch_size': BATCH_SIZE})

  classifier = tf.estimator.Estimator(
    model_fn=model_fn, config=config, params= hparams)

  train_spec = tf.estimator.TrainSpec(
    input_fn= lambda: input_fn(
      mode= tf.estimator.ModeKeys.TRAIN,
      num_epochs= NUM_EPOCHS,
      batch_size= BATCH_SIZE
    ),
    hooks= None,
  )

  eval_spec = tf.estimator.EvalSpec(
    input_fn= lambda: input_fn(
      mode= tf.estimator.ModeKeys.EVAL,
      num_epochs= 1,
      batch_size= 10000
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

import shutil
import math
import multiprocessing
from datetime import datetime
import itertools

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import data
from tensorflow.python.feature_column import feature_column


print(tf.__version__)


# Data pipeline input function
def csv_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL, 
                 skip_header_lines=0, 
                 num_epochs=None, 
                 batch_size=200):
    
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    
    num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1
    
    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")
    
    file_names = tf.matching_files(files_name_pattern)

    dataset = data.TextLineDataset(filenames=file_names)
    dataset = dataset.skip(skip_header_lines)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda csv_row: parse_csv_row(csv_row),  num_parallel_calls=num_threads)
    
    if PROCESS_FEATURES:
        dataset = dataset.map(lambda features, target: (process_features(features), target),  
                              num_parallel_calls=num_threads)
    
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    
    features, target = iterator.get_next()
    return features, target

# Parsing
def parse_csv_row(csv_row):
    
    columns = tf.decode_csv(csv_row, record_defaults=HEADER_DEFAULTS)
    features = dict(zip(HEADER, columns))
    
    for column in UNUSED_FEATURE_NAMES:
        features.pop(column)
    
    target = features.pop(TARGET_NAME)

    return features, target

# Preprocessing
def process_features(features):
    
    features['CRIM'] = tf.log(features['CRIM']+0.01)
    features['B'] = tf.clip_by_value(features['B'], clip_value_min=300, clip_value_max=500)
    
    return features

# Create Feature Columns
def get_feature_columns(hparams):
    
    numeric_columns = [
        tf.feature_column.numeric_column(feature_name) for feature_name in NUMERIC_FEATURE_NAMES]
    


    indicator_columns = [
        tf.feature_column.indicator_column(
           tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1]))
         for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()]
        
    return numeric_columns + indicator_columns

# Define model function
def model_fn(features, labels, mode, params, config):
    feature_columns = get_feature_columns(params)
    # Create first "numerical" layer by concatenating the features
    # transformed according to features_columns
    input_layer = feature_column.input_layer(features, feature_columns)
    
    # Defining the tf.keras.Model
    
    # The first step is defining a tf.keras.Input that matches
    # the dimension on input_layer
    input_layer_dimension = input_layer.shape.as_list()[1]
    inputs = tf.keras.Input(shape=(input_layer_dimension, ))
    # The second step is defining the hidden layers
    x = tf.keras.layers.Dense(params.hidden_units[0], activation=tf.nn.relu)(inputs)
    for layer_size in params.hidden_units[1:]:
        x = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)(x)
      
    # The output layer has size 1 because we are solving a regression problem
    outputs = tf.keras.layers.Dense(1)(x)
    # The final step is defining the tf.keras.Model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Now that we have our model we can compute the value of the logits 
    logits = model(input_layer)

    # passing in a parameter for gradient accumulation
    gradient_accumulation_multiplier= params.gradient_accumulation_multiplier
    
    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""

        global_step = tf.train.get_global_step()

        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        accum_grads = [tf.Variable(tf.zeros_like(t_var.initialized_value()), trainable=False) for t_var in tvars]

        optimizer = tf.train.AdamOptimizer()

        def apply_accumulated_gradients(accum_grads, grads, tvars):
            accum_op= tf.group([accum_grad.assign_add(grad) for (accum_grad, grad) in zip(accum_grads, grads)])
            with tf.control_dependencies([accum_op]):
                normalized_accum_grads = [1.0*accum_grad/gradient_accumulation_multiplier for accum_grad in accum_grads]
                # global_step is not incremented inside optimizer.apply_gradients
                minimize_op= optimizer.apply_gradients(zip(normalized_accum_grads, tvars), global_step = None)
                with tf.control_dependencies([minimize_op]):
                    zero_op= tf.group([accum_grad.assign(tf.zeros_like(accum_grad)) for accum_grad in accum_grads])
            return zero_op

        # Create training operation
        train_op = tf.cond(tf.math.equal(global_step % gradient_accumulation_multiplier, 0),
            lambda: apply_accumulated_gradients(accum_grads, grads, tvars),
            lambda: tf.group([accum_grad.assign_add(grad) for (accum_grad, grad) in zip(accum_grads, grads)])
        )

        # global_step is incremented here, regardless of the tf.cond branch
        train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
        return train_op
  
    
    head = tf.contrib.estimator.regression_head(
            label_dimension=1,
            name='regression_head'
        )
    
    return head.create_estimator_spec(
            features,
            mode,
            logits,
            labels=labels,
            train_op_fn=_train_op_fn
        )

# Define new metrics
def metric_fn(labels, predictions):

    metrics = {}

    pred_values = predictions['predictions']

    metrics["mae"] = tf.metrics.mean_absolute_error(labels, pred_values)
    metrics["rmse"] = tf.metrics.root_mean_squared_error(labels, pred_values)

    return metrics

# Define the estimator
def create_estimator(run_config, hparams):
    
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, 
        config=run_config,
        params=hparams
    )
    
    
    estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)
    
    return estimator

if __name__ == "__main__":
    # Settings
    # 
    print("############################################################################################")
    print("SETTINGS")
    MODEL_NAME = 'housing-price-model-01'

    DATA_FILE = 'data/housingdata.csv'

    TRAIN_DATA_FILES_PATTERN = 'data/housing-train-01.csv'
    TEST_DATA_FILES_PATTERN = 'data/housing-test-01.csv'

    RESUME_TRAINING = False
    PROCESS_FEATURES = True
    MULTI_THREADING = True

    # Define Dataset Metadata
    #  
    HEADER = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    HEADER_DEFAULTS = [[0.0],[0.0],[0.0],['NA'],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]

    NUMERIC_FEATURE_NAMES = ['CRIM', 'ZN','INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'CHAS':['0', '1']}
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())

    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

    TARGET_NAME = 'MEDV'

    UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})

    print("Header: {}".format(HEADER))
    print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
    print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
    print("Target: {}".format(TARGET_NAME))
    print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))

    # Load and analyse dataset
    housing_dataset = pd.read_csv(DATA_FILE, header=None, names=HEADER )
    # housing_dataset.head()

    # Prepare Training and Test Sets
    DATA_SIZE = len(housing_dataset)

    print("Dataset size: {}".format(DATA_SIZE))

    train_data = housing_dataset.sample(frac=0.70, random_state = 19830610)
    test_data = housing_dataset[~housing_dataset.index.isin(train_data.index)]

    TRAIN_DATA_SIZE = len(train_data)
    TEST_DATA_SIZE = len(test_data)

    print("Train set size: {}".format(TRAIN_DATA_SIZE))
    print("Test set size: {}".format(TEST_DATA_SIZE))
    print("")

    # Save Training and Test Sets
    train_data.to_csv(path_or_buf="data/housing-train-01.csv", header=False, index=False)
    test_data.to_csv(path_or_buf="data/housing-test-01.csv", header=False, index=False)

    features, target = csv_input_fn(files_name_pattern="")
    print("Features in CSV: {}".format(list(features.keys())))
    print("Target in CSV: {}".format(target))

    feature_columns = get_feature_columns(tf.contrib.training.HParams(num_buckets=5,embedding_size=3))
    print("Feature Columns: {}".format(feature_columns))

    # Run Experiment: 
    # Set HParam and RunConfig
    TRAIN_SIZE = TRAIN_DATA_SIZE
    NUM_EPOCHS = 10000
    BATCH_SIZE = 59 #177
    EVAL_AFTER_SEC = 30
    TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS

    hparams  = tf.contrib.training.HParams(
        num_epochs = NUM_EPOCHS,
        batch_size = BATCH_SIZE,
        gradient_accumulation_multiplier = 3,
        hidden_units=[16, 8, 4],
        max_steps = TOTAL_STEPS
    )

    model_dir = 'trained_models/{}'.format(MODEL_NAME)

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=1000,
        tf_random_seed=19830610,
        model_dir=model_dir
    )

    print(hparams)
    print("Model Directory:", run_config.model_dir)
    print("")
    print("Dataset Size:", TRAIN_SIZE)
    print("Batch Size:", BATCH_SIZE)
    print("Steps per Epoch:",TRAIN_SIZE/BATCH_SIZE)
    print("Total Steps:", TOTAL_STEPS)
    print("That is 1 evaluation step after each",EVAL_AFTER_SEC," training seconds")

    # Define TrainSpec and EvalSpec
    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: csv_input_fn(
            TRAIN_DATA_FILES_PATTERN,
            mode = tf.estimator.ModeKeys.TRAIN,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.batch_size
        ),
        max_steps=hparams.max_steps,
        hooks=None
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: csv_input_fn(
            TRAIN_DATA_FILES_PATTERN,
            mode=tf.estimator.ModeKeys.EVAL,
            num_epochs=1,
            batch_size=hparams.batch_size,
                
        ),
        throttle_secs = EVAL_AFTER_SEC,
        steps=None
    )

    # Run Experiment via train_and_evaluate
    if not RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        print("Resuming training...") 

        
    tf.logging.set_verbosity(tf.logging.INFO)

    time_start = datetime.utcnow() 
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................") 

    estimator = create_estimator(run_config, hparams)

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec, 
        eval_spec=eval_spec
    )

    time_end = datetime.utcnow() 
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))

    # Evaluate the Model
    train_input_fn = lambda: csv_input_fn(files_name_pattern= TRAIN_DATA_FILES_PATTERN, 
                                        mode= tf.estimator.ModeKeys.EVAL,
                                        batch_size= TRAIN_DATA_SIZE)


    test_input_fn = lambda: csv_input_fn(files_name_pattern= TEST_DATA_FILES_PATTERN, 
                                        mode= tf.estimator.ModeKeys.EVAL,
                                        batch_size= TEST_DATA_SIZE)

    estimator = create_estimator(run_config, hparams)

    train_results = estimator.evaluate(input_fn=train_input_fn, steps=1)
    train_rmse = round(math.sqrt(train_results["rmse"]),5)
    print()
    print("############################################################################################")
    print("# Train RMSE: {} - {}".format(train_rmse, train_results))
    print("############################################################################################")

    test_results = estimator.evaluate(input_fn=test_input_fn, steps=1)
    test_rmse = round(math.sqrt(test_results["rmse"]),5)
    print()
    print("############################################################################################")
    print("# Test RMSE: {} - {}".format(test_rmse, test_results))
    print("############################################################################################")    

    # Prediction
    print("############################################################################################")
    print("PREDICTIONS")

    predict_input_fn = lambda: csv_input_fn(files_name_pattern= TEST_DATA_FILES_PATTERN, 
                                        mode= tf.estimator.ModeKeys.PREDICT,
                                        batch_size= 5)

    predictions = estimator.predict(input_fn=predict_input_fn)
    values = list(map(lambda item: item["predictions"][0],list(itertools.islice(predictions, 5))))
    print()
    print("Predicted Values: {}".format(values))    
    
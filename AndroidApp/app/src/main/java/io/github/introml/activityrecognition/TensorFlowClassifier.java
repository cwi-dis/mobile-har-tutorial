package io.github.introml.activityrecognition;

import android.content.Context;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class TensorFlowClassifier {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/ucd_keras_frozen3.pb";
//    private static final String MODEL_FILE = "file:///android_asset/frozen_har.pb";
    private static final String INPUT_NODE = "conv2d_1_input";
    private static final String[] OUTPUT_NODES = {"dense_3/Softmax"};
    private static final String OUTPUT_NODE = "dense_3/Softmax";


    private static final int N_CHANNELS = 1;
    private static final int N_FEATURES = 6;
    private static final int N_STEPS = 90;

//    private static final String[] labels = new String[]{"WalkForward","WalkLeft","WalkRight","WalkUp","WalkDown","RunForward", "JumpUp", "Sit", "Stand", "Sleep", "ElevatorUp", "ElevatorDown"};




    //
//    private static final String[] OUTPUT_NODES = {"y_"};
//    private static final String OUTPUT_NODE = "y_";
////    private static final long[] INPUT_SIZE = {1, 200, 3};
    private static final long[] INPUT_SIZE = {1, N_STEPS, N_FEATURES, N_CHANNELS};
    private static final int OUTPUT_SIZE = 12;

    public TensorFlowClassifier(final Context context) {
        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);
    }

    public float[] predictProbabilities(float[] data) {
        float[] result = new float[OUTPUT_SIZE];
        Log.v("result_tf_classify", String.valueOf(result));

        inferenceInterface.feed(INPUT_NODE, data, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, result);

        //['WalkForward','WalkLeft','WalkRight','WalkUp','WalkDown','RunForward', 'JumpUp', 'Sit', 'Stand', 'Sleep', 'ElevatorUp', 'ElevatorDown']
        return result;
    }
}

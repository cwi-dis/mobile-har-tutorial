package io.github.introml.activityrecognition;

import android.content.Context;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class TensorFlowClassifier {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/CNN_model.pb";
    private static final String INPUT_NODE = "conv2d_1_input:0";
    private static final String[] OUTPUT_NODES = {"dense_3/Softmax:0"};
    private static final String OUTPUT_NODE = "dense_3/Softmax:0";

    private static final int N_CHANNELS = 1;
    private static final int N_FEATURES = 3;
    private static final int N_STEPS = 90;

//    private static final String[] labels = new String[]{"WalkForward","WalkLeft","WalkRight","WalkUp","WalkDown","RunForward", "JumpUp", "Sit", "Stand", "Sleep", "ElevatorUp", "ElevatorDown"};

    // private static final long[] INPUT_SIZE = {1, 1, N_STEPS, N_FEATURES, N_CHANNELS}; // ConvLSTM
    private static final long[] INPUT_SIZE = {1, N_STEPS, N_FEATURES, N_CHANNELS}; // CNN

    private static final int OUTPUT_SIZE = 12;

    public TensorFlowClassifier(final Context context) {
        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);
    }

    public float[] predictProbabilities(float[] data) {
        float[] result = new float[OUTPUT_SIZE];

        inferenceInterface.feed(INPUT_NODE, data, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, result);

        return result;
    }
}

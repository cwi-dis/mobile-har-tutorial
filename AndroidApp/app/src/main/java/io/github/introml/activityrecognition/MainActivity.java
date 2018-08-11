package io.github.introml.activityrecognition;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.TextView;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;


public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener {

    private static final int N_SAMPLES = 90;
    private static List<Float> x;
    private static List<Float> y;
    private static List<Float> z;
    private static List<Float> gyr_x;
    private static List<Float> gyr_y;
    private static List<Float> gyr_z;


    private TextView walkForwardTextView;
    private TextView walkLeftTextView;
    private TextView walkRightTextView;
    private TextView walkUpTextView;
    private TextView walkDownTextView;
    private TextView runForwardTextView;
    private TextView jumpUpTextView;
    private TextView sitTextView;
    private TextView standTextView;
    private TextView sleepTextView;
    private TextView elevatorUpTextView;
    private TextView elevatorDownTextView;
    private TextToSpeech textToSpeech;
    private float[] results;
    private TensorFlowClassifier classifier;

    //    private String[] labels = {"Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"};
    private String[] labels = {"WalkForward","WalkLeft","WalkRight","WalkUp","WalkDown","RunForward", "JumpUp", "Sit", "Stand", "Sleep", "ElevatorUp", "ElevatorDown"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        x = new ArrayList<>();
        y = new ArrayList<>();
        z = new ArrayList<>();
        gyr_x = new ArrayList<>();
        gyr_y = new ArrayList<>();
        gyr_z = new ArrayList<>();

        walkForwardTextView = (TextView) findViewById(R.id.walkforward_prob);
        walkLeftTextView = (TextView) findViewById(R.id.walkleft_prob);
        walkRightTextView = (TextView) findViewById(R.id.walkright_prob);
        walkUpTextView = (TextView) findViewById(R.id.walkup_prob);
        walkDownTextView = (TextView) findViewById(R.id.walkdown_prob);
        runForwardTextView = (TextView) findViewById(R.id.runforward_prob);
        jumpUpTextView = (TextView) findViewById(R.id.jump_prob);
        sitTextView = (TextView) findViewById(R.id.sit_prob);
        standTextView = (TextView) findViewById(R.id.stand_prob);
        sleepTextView = (TextView) findViewById(R.id.sleep_prob);
        elevatorUpTextView = (TextView) findViewById(R.id.elevatorup_prob);
        elevatorDownTextView = (TextView) findViewById(R.id.elevatordown_prob);

        classifier = new TensorFlowClassifier(getApplicationContext());

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);
    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (results == null || results.length == 0) {
                    return;
                }
                float max = -1;
                int idx = -1;
                for (int i = 0; i < results.length; i++) {
                    if (results[i] > max) {
                        idx = i;
                        max = results[i];
                    }
                }

                textToSpeech.speak(labels[idx], TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
            }
        }, 3000, 5000);
    }

    protected void onPause() {
        getSensorManager().unregisterListener(this);
        super.onPause();
    }

    protected void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME);
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_GAME);

    }

    @Override
    public void onSensorChanged(SensorEvent event) {


        synchronized (this) {
            float[] accels = new float[0];
            float[] gyrs = new float[0];
            switch (event.sensor.getType()) {
                case Sensor.TYPE_ACCELEROMETER:
                    x.add(event.values[0]);
                    y.add(event.values[1]);
                    z.add(event.values[2]);
//                    arrayCopy(event.values, gravity);
                    accels = event.values.clone();
                    break;

                case Sensor.TYPE_GYROSCOPE:
                    gyr_x.add(event.values[0]);
                    gyr_y.add(event.values[1]);
                    gyr_z.add(event.values[2]);
                    gyrs = event.values.clone();
                    break;
            }

            ArrayList<float[]> tmp2 = new ArrayList<>();
            List<List<Float>> tmp = new ArrayList<>();


            if (!x.isEmpty() && !y.isEmpty()){
                tmp.add(x.subList(0,1));
                tmp.add(y.subList(0,1));

            }

//        tmp.subList(0,89);

//            tmp.add(y);
//            tmp.add(gyr_x);
//            tmp.add(gyr_y);


            Collections.addAll(tmp2, gyrs);
            Collections.addAll(tmp2, accels);

            Log.v("x,y,gyr_x, gyr_y", Arrays.toString(new List[]{tmp}));
//            Log.v("all", Arrays.toString(new ArrayList[]{tmp2}));

        activityPrediction();

//            Log.v("acc_x", String.valueOf(x));
//            Log.v("gyr_x", String.valueOf(gyr_x));
//            Log.v("gyr_z", String.valueOf(gyr_z));

    }
}

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }


//    Sanity check //

//      int n = 90;
//      for (int i=1; i<=n; i++) {
//          x.add((float) i);
////          y.add((float) i);
////          z.add((float) i);
////          gyr_x.add((float) i);
////          gyr_y.add((float) i);
////          gyr_z.add((float) i);
//
//      }
//
//      for (int i=91; i<=180; i++) {
////          x.add((float) i);
////          y.add((float) i);
////          z.add((float) i);
//          gyr_x.add((float) i);
////          gyr_y.add((float) i);
////          gyr_z.add((float) i);
//
//      }


    private void activityPrediction() {


        if (x.size() == N_SAMPLES && y.size() == N_SAMPLES && z.size() == N_SAMPLES && gyr_x.size() == N_SAMPLES && gyr_y.size() == N_SAMPLES && gyr_z.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();

            data.addAll(x);
            data.addAll(y);
            data.addAll(z);
            data.addAll(gyr_x);
            data.addAll(gyr_y);
            data.addAll(gyr_z);

            Log.v("data", Arrays.toString(toFloatArray(data)));


            // TODO:
            // Need data in following format
            // shape: (1, 90, 6, 1) #(observations, timesteps/n_samples, features (acc + gyro), channels).
//            [[[ 1.07006741e+00]
//              [ 4.07870710e-01]
//              [-3.87284398e-01]
//              [ 3.16722336e+01]
//              [ 7.24022865e+00]
//              [-6.91453934e+00]]
//              ...
//             [[ 1.27952909e+00]
//              [ 3.82536739e-01]
//              [-3.43362540e-01]
//              [ 2.96580734e+01]
//              [ 7.24022865e+00]
//              [-1.61796761e+01]]] # each data point / observation has 90 samples with 6 features
//
            results = classifier.predictProbabilities(toFloatArray(data));

            Log.v("results", String.valueOf(results));


            walkForwardTextView.setText(Float.toString(round(results[0], 2)));
            walkLeftTextView.setText(Float.toString(round(results[1], 2)));
            walkRightTextView.setText(Float.toString(round(results[2], 2)));
            walkUpTextView.setText(Float.toString(round(results[3], 2)));
            walkDownTextView.setText(Float.toString(round(results[4], 2)));
            runForwardTextView.setText(Float.toString(round(results[5], 2)));
            jumpUpTextView.setText(Float.toString(round(results[6], 2)));
            sitTextView.setText(Float.toString(round(results[7], 2)));
            standTextView.setText(Float.toString(round(results[8], 2)));
            sleepTextView.setText(Float.toString(round(results[9], 2)));
            elevatorUpTextView.setText(Float.toString(round(results[10], 2)));
            elevatorDownTextView.setText(Float.toString(round(results[11], 2)));

            x.clear();
            y.clear();
            z.clear();
            gyr_x.clear();
            gyr_y.clear();
            gyr_z.clear();
        }
    }

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }

}
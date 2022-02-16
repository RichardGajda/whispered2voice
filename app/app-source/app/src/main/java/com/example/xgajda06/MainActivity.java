package com.example.xgajda06;

import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import static android.Manifest.permission.RECORD_AUDIO;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

public class MainActivity extends AppCompatActivity {

    // INIT ALL
    private TextView rec_btn, stop_rec_btn, play_conv, stop_play_conv, textView, btnMale, btnFemale;

    // INIT ALL MEDIA VARIABLES

    private MediaRecorder mediaRecorder;
    private MediaPlayer mediaPlayer;
    private static String media_file_name = null;
    public static final int REQUEST_AUDIO_PERMISSION_CODE = 1;

    // AUXILLIARY VARIABLES

    private String rec_file;
    private String aux_string;
    private File file;
    private String gender = "none";


    //INIT ACTIVITY SCREEN, ACCESS UI ELEMENTS
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.status);
        rec_btn = findViewById(R.id.btnRecord);
        stop_rec_btn = findViewById(R.id.btnStop);
        play_conv = findViewById(R.id.btnPlay);
        stop_play_conv = findViewById(R.id.btnStopPlay);
        btnMale = findViewById(R.id.btnMale);
        btnFemale = findViewById(R.id.btnFemale);

        //INIT BUTTONS, DISABLE THEM UNTIL GENDER IS SELECTED

        rec_btn.setEnabled(false);
        stop_rec_btn.setEnabled(false);
        play_conv.setEnabled(false);
        stop_play_conv.setEnabled(false);

        rec_btn.setBackgroundColor(getResources().getColor(R.color.gray));
        stop_rec_btn.setBackgroundColor(getResources().getColor(R.color.gray));
        play_conv.setBackgroundColor(getResources().getColor(R.color.gray));
        stop_play_conv.setBackgroundColor(getResources().getColor(R.color.gray));

        textView.setText(R.string.select_gender);

        // SELECTING MALE HANDLER

        btnMale.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                gender = "male";
                btnMale.setBackgroundColor(getResources().getColor(R.color.orange_default));
                btnFemale.setBackgroundColor(getResources().getColor(R.color.gray));
                textView.setText(R.string.male_selected);

                rec_btn.setEnabled(true);
                rec_btn.setBackgroundColor(getResources().getColor(R.color.orange_default));
                play_conv.setEnabled(false);
                stop_play_conv.setEnabled(false);

            }
        });

        // SELECTING FEMALE HANDLER

        btnFemale.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                gender = "female";
                btnMale.setBackgroundColor(getResources().getColor(R.color.gray));
                btnFemale.setBackgroundColor(getResources().getColor(R.color.orange_default));
                textView.setText(R.string.female_selected);

                rec_btn.setEnabled(true);
                rec_btn.setBackgroundColor(getResources().getColor(R.color.orange_default));
                play_conv.setEnabled(false);
                stop_play_conv.setEnabled(false);


            }
        });

        // REC BUTTON

        rec_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startRecording();
            }
        });

        // PAUSE BUTTON, UPON WHICH THE HTTP COMMUNICATION IS DONE.

        stop_rec_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                pauseRecording();

                //RECORDING STOPPED, BUILD A REQUEST AND START STREAMING TO SERVER

                OkHttpClient okHttpClient=new OkHttpClient.Builder().connectTimeout(120, TimeUnit.SECONDS).writeTimeout(120, TimeUnit.SECONDS).readTimeout(120, TimeUnit.SECONDS).build();
                MultipartBody.Builder mMultipartBody = new MultipartBody.Builder();
                mMultipartBody.setType(MultipartBody.FORM);
                File f = new File(media_file_name);
                mMultipartBody.addFormDataPart("file", rec_file, RequestBody.create(MediaType.parse("application/octet-stream"), f));
                mMultipartBody.addFormDataPart("gender", gender);
                RequestBody mRequestBody = mMultipartBody.build();
                aux_string = "Converting, please wait.";
                textView.setText(aux_string);

                //SENDING REQUEST

                Request request = new Request.Builder().url("https://app-test-xgajda06.herokuapp.com/post").post(mRequestBody).build();
                okHttpClient.newCall(request).enqueue(new Callback(){

                    //RESPONSE HANDLER
                    @Override
                    public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {

                        aux_string = response.body().string();

                        //IF CONVERSION FAILS (THIS IS MOSTLY BECAUSE REQUEST TIMED OUT ON SERVER, THIS CAN OCCUR RANDOMLY ON NON-COMMERCIAL SERVERS.

                        if (!aux_string.equals("CONVERSION OK")){
                            //conversion failed mid
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    aux_string = "Conversion failed, try again.";
                                    btnFemale.setEnabled(true);
                                    btnMale.setEnabled(true);
                                    btnMale.setBackgroundColor(getResources().getColor(R.color.orange_default));
                                    btnFemale.setBackgroundColor(getResources().getColor(R.color.orange_default));

                                }
                            });
                        }

                        //PRINT CONVERSION OK

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                textView.setText(aux_string);

                            }
                        });

                        //CONVERSION OK, SO NOW REQUEST THE CONVERTED FILE AND SAVE IT

                        Request request_file = new Request.Builder().url("https://app-test-xgajda06.herokuapp.com/get_conv").build();
                        try (Response response1 = okHttpClient.newCall(request_file).execute()){
                            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response1);

                            //BUILD THE OUTPUT FILE FROM THE STREAM FROM SERVER

                            FileOutputStream outputStream;
                            String name="aux.wav";
                            File downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS); ;
                            file = new File(downloads, name);
                            media_file_name = downloads + "/" + name;
                            outputStream = new FileOutputStream(file);
                            outputStream.write(response1.body().bytes());

                            Log.d("Path", Environment.getDataDirectory().toString());
                            outputStream.close();
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {

                                    //DOWNLOAD SUCCESSFUL, ENABLE PLAYBACK OR RECORDING NEW UTTERANCE

                                    textView.setText(R.string.dlcomplete);
                                    //statusTV.setText(mFileName);
                                    play_conv.setEnabled(true);
                                    play_conv.setBackgroundColor(getResources().getColor(R.color.orange_default));
                                    btnFemale.setEnabled(true);
                                    btnMale.setEnabled(true);
                                    btnMale.setBackgroundColor(getResources().getColor(R.color.orange_default));
                                    btnFemale.setBackgroundColor(getResources().getColor(R.color.orange_default));

                                }
                            });
                        }
                    }

                    @Override
                    public void onFailure(@NotNull Call call, @NotNull IOException e) {
                        Looper.prepare();
                        Toast.makeText(getApplicationContext(), "Network not found", Toast.LENGTH_LONG).show();
                    }
                });
            }
        });
        play_conv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                playAudio();
            }
        });
        stop_play_conv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                pausePlaying();
            }
        });
    }

    private void startRecording() {

        //PERMISSION CHECK UPON STARTING THE APP FOR THE FIRST TIME, IF GRANTED:

        if (CheckPermissions()) {

            stop_rec_btn.setBackgroundColor(getResources().getColor(R.color.orange_def));
            rec_btn.setBackgroundColor(getResources().getColor(R.color.gray));
            play_conv.setBackgroundColor(getResources().getColor(R.color.gray));
            stop_play_conv.setBackgroundColor(getResources().getColor(R.color.gray));

            //INIT FILENAME

            media_file_name = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath(); ;
            media_file_name += "/AudioRecording.wav";
            rec_file = "AudioRecording.wav";

            //INIT RECORDER

            mediaRecorder = new MediaRecorder();

            //SOURCE = INTEGRATED PHONE MICROPHONE

            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);

            //SET OUTPUT FORMAT, CONTAINER, SAMPLING FREQUENCY AND BITRATE.
            //DEFAULT FORMAT WAS 3GP OR AMR, BUT WAV IS PREFERABLE FOR OUR PURPOSES

            mediaRecorder.setOutputFormat(AudioFormat.ENCODING_PCM_16BIT);
            mediaRecorder.setAudioChannels(1);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setAudioEncodingBitRate(128000);
            mediaRecorder.setAudioSamplingRate(44100);

            //LINK CREATED OUTPUT FILE TO THE MEDIA RECORDER OUTPUT

            mediaRecorder.setOutputFile(media_file_name);

            //PREPARE AND START RECORDING

            try {
                mediaRecorder.prepare();
            } catch (IOException e) {
                Log.e("TAG", "prepare() failed");
            }
            mediaRecorder.start();
            textView.setText(R.string.rec_start);

            //ENSURE RECORDING CAN ONLY BE STOPPED BY STOP RECORDING BUTTON

            rec_btn.setEnabled(false);
            stop_rec_btn.setEnabled(true);
            btnFemale.setEnabled(false);
            btnMale.setEnabled(false);
            btnMale.setBackgroundColor(getResources().getColor(R.color.gray));
            btnFemale.setBackgroundColor(getResources().getColor(R.color.gray));
        } else {

            //IF PERMISSIONS NOT GRANTED

            RequestPermissions();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {

        //CHANGE PERMISSION CODES ACCORDING TO REQUEST RESULTS.

        switch (requestCode) {
            case REQUEST_AUDIO_PERMISSION_CODE:
                if (grantResults.length > 0) {
                    boolean permissionToRecord = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean permissionToStore = grantResults[1] == PackageManager.PERMISSION_GRANTED;
                    if (permissionToRecord && permissionToStore) {
                        Toast.makeText(getApplicationContext(), "Permission Granted", Toast.LENGTH_LONG).show();
                    } else {
                        Toast.makeText(getApplicationContext(), "Permission Denied", Toast.LENGTH_LONG).show();
                    }
                }
                break;
        }
    }

    public boolean CheckPermissions() {

        //PRETTY SELF-EXPLANATORY, CHECK PERMISSIONS

        int result = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int result1 = ContextCompat.checkSelfPermission(getApplicationContext(), RECORD_AUDIO);
        return result == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED;
    }

    private void RequestPermissions() {

        //REQUEST PERMISSIONS

        ActivityCompat.requestPermissions(MainActivity.this, new String[]{RECORD_AUDIO, WRITE_EXTERNAL_STORAGE}, REQUEST_AUDIO_PERMISSION_CODE);
    }


    public void playAudio() {

        //ONLY STOP PLAY BUTTON CAN INTERRUPT THE PLAYBACK

        stop_rec_btn.setBackgroundColor(getResources().getColor(R.color.gray));
        rec_btn.setBackgroundColor(getResources().getColor(R.color.orange_def));
        play_conv.setBackgroundColor(getResources().getColor(R.color.gray));
        stop_play_conv.setBackgroundColor(getResources().getColor(R.color.orange_def));

        //INITIALIZE MEDIA PLAYER WHICH WILL PLAY THE RECORDING

        mediaPlayer = new MediaPlayer();
        try {

            //PARSE THE MEDIA FILE

            mediaPlayer.setDataSource(media_file_name);

            //PREPARE

            mediaPlayer.prepare();

            //START PLAYING

            mediaPlayer.start();
            stop_play_conv.setEnabled(true);
            textView.setText(R.string.play_conv);
        } catch (IOException e) {
            Log.e("TAG", "prepare() failed");
        }
    }

    public void pauseRecording() {

        //BUTTON HANDLER

        stop_rec_btn.setBackgroundColor(getResources().getColor(R.color.gray));
        rec_btn.setBackgroundColor(getResources().getColor(R.color.gray));
        play_conv.setBackgroundColor(getResources().getColor(R.color.gray));
        stop_play_conv.setBackgroundColor(getResources().getColor(R.color.gray));

        //STOP THE STREAM INTO OUTPUT FILE

        mediaRecorder.stop();

        //RELEASING THE MEDIA RECORDER CLASS, UPON THIS, THE FILE IS CREATED

        mediaRecorder.release();
        mediaRecorder = null;
        textView.setText(R.string.rec_stop);
        stop_rec_btn.setEnabled(false);
    }

    public void pausePlaying() {

        //RELEASE THE MEDIA PLAYER

        mediaPlayer.release();
        mediaPlayer = null;

        //BUTTON HANDLING, UPON FINISHING THE PLAYBACK, USER CAN RECORD A NEW UTTERANCE.

        stop_rec_btn.setBackgroundColor(getResources().getColor(R.color.gray));
        rec_btn.setBackgroundColor(getResources().getColor(R.color.orange_def));
        play_conv.setBackgroundColor(getResources().getColor(R.color.orange_def));
        stop_play_conv.setBackgroundColor(getResources().getColor(R.color.gray));
        textView.setText(R.string.play_stop);
        rec_btn.setEnabled(true);
        stop_play_conv.setEnabled(false);
    }
}

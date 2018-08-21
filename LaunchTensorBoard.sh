#! /bin/bash


pkill "python";
echo Starting to launch Tensorboard;

sleep 3;

tensorboard --logdir=log &
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --enable-speech-input http://localhost:6006 &
open -a Google\ Chrome

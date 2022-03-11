#!/bin/bash
# File: data_handler_test.sh
# Brief: Test the functions used to create the ML input data.
cd ~/Documents/Programming_2/Metal_Mixer
python -m Code.Data_Handling.audioHandlerTest
python -m Code.Data_Handling.instrumentDataParserTest
python -m Code.Data_Handling.instrumentMixerTest
python -m Code.Data_Handling.spectrogramMLDataGeneratorTest
python -m Code.Data_Handling.trueSourceGeneratorTest

#pragma once

#include "pch.h"
#include "../include/raylib.h"
#include "UI.h"
#include "NeuralNetwork.h"
#include "Math.h"

namespace NN {

class Application {

public:
  struct ApplicationSpec {
    std::string _title;
    u32 _windowWidth;
    u32 _windowHeight;
    u32 _fps;

    ApplicationSpec(const std::string &title = "Engine", u32 windowWidth = 800, u32 windowHeight = 600, u32 fps = 60) :
        _title(title), _windowHeight(windowHeight), _windowWidth(windowWidth), _fps(fps) {};
  };

public:
  UI ui;

  NN::NeuralNetwork nn;

  Application(const ApplicationSpec &spec);

  void run();
  void drawDigitsTerminal(std::vector<f32> data);
  void loadImage(u32 index);
  u8 getLabel(std::vector<f32> &data, u64 index);
  void onKeyPressed();

private:
  struct ApplicationData {
    std::string title;
    u32 width;
    u32 height;
    u32 fps;
  };

  ApplicationData _data;
};

}  // namespace NN

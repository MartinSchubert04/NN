#pragma once

#include "pch.h"
#include "../include/raylib.h"
#include "UI.h"
#include "NeuralNetwork.h"
#include "Math.h"
#include <thread>
#include <atomic>

namespace NN {

class Application {

public:
  struct ApplicationSpec {
    std::string _title;
    f32 _windowWidth;
    f32 _windowHeight;
    f32 _fps;

    ApplicationSpec(const std::string &title = "Engine", f32 windowWidth = 800, f32 windowHeight = 600,
                    f32 fps = 60.0f) :
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
  void copyImageToGrid(u32 index);

private:
  struct ApplicationData {
    std::string title;
    f32 width;
    f32 height;
    f32 fps;
  };

  ApplicationData _data;
};

}  // namespace NN

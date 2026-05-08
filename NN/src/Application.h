#pragma once

#include "pch.h"
#include "../include/raylib.h"

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
  Application(const ApplicationSpec &spec);

  void run();
  void drawDigits(std::vector<f32> data);

private:
  struct ApplicationData {
    std::string title;
    u32 width;
    u32 height;
    u32 fps;
  };

  ApplicationData _data;
};

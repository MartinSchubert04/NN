#include "Application.h"

Application::Application(const ApplicationSpec &spec) {
  _data.title = spec._title;
  _data.width = spec._windowWidth;
  _data.height = spec._windowHeight;
  _data.fps = spec._fps;

  InitWindow(_data.width, _data.height, _data.title.c_str());
}

void Application::run() {

  while (!WindowShouldClose()) {
    SetWindowTitle(TextFormat("%s | FPS: %d", _data.title.c_str(), GetFPS()));

    BeginDrawing();

    ClearBackground(BLACK);

    DrawText("Hello", 160, 150, 20, LIGHTGRAY);

    EndDrawing();
  }
}

void Application::drawDigits(std::vector<f32> data) {
  for (u32 y{0}; y < 28; y++) {
    for (u32 x{0}; x < 28; x++) {
      f32 num = data[x + y * 28];
      u32 col = 232 + (u32)(num * 23);
      printf("\x1b[48;5;%dm  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

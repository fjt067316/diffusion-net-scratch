#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <chrono>
#include <future>
#include <string>
#include <semaphore.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <random>

using namespace std;
namespace fs = filesystem;

#define EPOCHS 100

// Define a simple image struct
struct Image {
    std::string filename;
    // Other image data can be added here
};

vector<vector<vector<uint8_t>>> readImage(const string& filePath) {
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR); // Read the image

    if (image.empty()) {
        cerr << "Error: Unable to read the image." << endl;
        return {};
    }

    // Extract pixel values
    vector<vector<vector<uint8_t>>> pixels(image.rows, vector<vector<uint8_t>>(image.cols, vector<uint8_t>(3)));
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b intensity = image.at<cv::Vec3b>(i, j);
            pixels[i][j][0] = intensity[0]; // Blue
            pixels[i][j][1] = intensity[1]; // Green
            pixels[i][j][2] = intensity[2]; // Red
        }
    }

    return pixels;
}

std::queue<vector<vector<vector<uint8_t>>>> imageQueue;
std::mutex queueMutex;
sem_t sem;

// Function to simulate writing image files to the queue
void writeImageFiles() {
    string path = "./images/";
    for(int j=0; j<EPOCHS; j++){
        fs::directory_iterator b(path), e;
        vector<fs::path> image_paths(b, e);
        int n_images = image_paths.size();

        // shuffle data
        auto rng = std::default_random_engine {};
        shuffle(begin(image_paths), end(image_paths), rng);

        for(int i=0; i < n_images; i++ ){

            fs::path next_image = image_paths.pop_back();
            vector<vector<vector<uint8_t>>> image = readImage(next_image.u8string());
            // Lock the queue to safely push data
            queueMutex.lock();
            imageQueue.push(image);
            queueMutex.unlock();
            std::cout << "Image file " << next_image.filename() << " added to the queue." << std::endl;
        }
    }
}

// Function to simulate reading image files from the queue
vector<vector<vector<uint8_t>>>* readImageFiles() {

    while (true) {
        // Lock the queue to safely pop data
        queueMutex.lock();
        if (!imageQueue.empty()) {
            vector<vector<vector<uint8_t>>> img = imageQueue.front();
            imageQueue.pop();
            queueMutex.unlock();

            return &img;
        } else {
            queueMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep for a short time if queue is empty
        }
    }
}

// in order to avoid the program waiting on image loads
// one thread will grab images into a n=3 queue while one thread pulls from queue and runs model
int main() {

    vector<vector<vector<uint8_t>>> out = readImage("images/test.jpg");
    // for (int i = 0; i < out.size(); ++i) {
    //     for (int j = 0; j < out[i].size(); ++j) {
    //         cout << "Pixel at (" << i << ", " << j << "): ";
    //         cout << "B: " << static_cast<int>(out[i][j][0]) << ", ";
    //         cout << "G: " << static_cast<int>(out[i][j][1]) << ", ";
    //         cout << "R: " << static_cast<int>(out[i][j][2]) << endl;
    //     }
    // }

    return 0;

    // sem_init(&sem, 0, 0);
    // promise<Image*> promise;
    // future<Image*> future = promise.get_future();

    // // Create threads for writing and reading image files
    // thread writerThread(writeImageFiles);
    // thread readerThread(readImageFiles, move(promise));

    // // Join the threads with the main thread
    // writerThread.join();
    // readerThread.join();

    // // get file
    // while(true){
    //     Image* image = future.get();




    //     free(image);

    // }

    // return 0;
}

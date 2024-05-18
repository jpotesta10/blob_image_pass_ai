void detectObjects(cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0,0,0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    for (auto& output : outs) {
        for (int i = 0; i < output.rows; i++) {
            float confidence = output.at<float>(i, 4);
            if (confidence > 0.5) {
                int classId = -1;
                float maxClassScore = -1;
                for (int j = 5; j < output.cols; j++) {
                    float classScore = output.at<float>(i, j);
                    if (classScore > maxClassScore) {
                        maxClassScore = classScore;
                        classId = j - 5;
                    }
                }
                if (classId >= 0) {
                    int centerX = static_cast<int>(output.at<float>(i, 0) * frame.cols);
                    int centerY = static_cast<int>(output.at<float>(i, 1) * frame.rows);
                    int width = static_cast<int>(output.at<float>(i, 2) * frame.cols);
                    int height = static_cast<int>(output.at<float>(i, 3) * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 2);
                    std::string label = std::to_string(classId) + ": " + std::to_string(confidence);
                    cv::putText(frame, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }
            }
        }
    }
}

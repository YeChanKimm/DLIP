# Anomaly Detection in Metal Nuts Using Reconstruction Error of Autoencoder



**Date:**  2025-May-21

**Author:**  Yechan Kim  22100153 / Haeun Kim 22100224

**Github:** https://github.com/HaeunKim2/DLIP_FINAL

**Demo Video:** https://youtu.be/Ng-ShvqlCxg

---

# Introduction

  In modern industrial environments, quality assurance is essential in ensuring the reliability and safety of manufactured components. Among these, metal nuts are essential fasteners used in nearly all mechanical and electronic assemblies.  Detecting defects in metal nuts, such as cracks, deformities, or surface damage, is vital to prevent mechanical failures and maintain product integrity. However, it is not easy for a person to manually check whether a small nut has any defects, and the process also takes a long time.

  This project aims to develop a visual anomaly detection system for identifying damaged metal nuts using deep learning-based autoencoding techniques. Leveraging the metal nuts from MVTec Anomaly Detection (AD) dataset, the model will be trained in an unsupervised manner using only normal samples. The system is designed to reconstruct input images through a convolutional autoencoder, and compute reconstruction errors to flag anomalous inputs.

By automating metal nuts defect inspection, the proposed method contributes to the advancement of smart factory technologies, where real-time and accurate defect detection enhance manufacturing efficiency and safety.

**Goal**: Detect damaged metal nuts by training an unsupervised autoencoder model on normal images from the MVTec AD dataset.

<img src="https://i.imgur.com/Iontl7y.png" style="zoom:80%;" />





# Problem Statement

  Detecting defects in metal nuts, such as cracks, deformities, or surface damage, is vital to prevent mechanical failures and maintain product integrity. However, it is not easy for a person to manually check whether a small nut has any defects, and the process also takes a long time.

**Expected Outcome: ** The project aims to deliver a real-time anomaly detection system that identifies damaged Metal nuts with high reliability. 

**Evaluation:** The trained autoencoder should achieve a accuracy of at least **75%** for defective metal nuts.



------

# Requirement

## Hardware

- Conveyor Belt

- Machine Vision Camera: e2V EV76C560 (I.I.LAB camera_014)

- Camera Lens (EDMUND OPTICS)

- Camera Cable

- Ring Flash

- Metal Profile

- Camera-Bracket Connector (stl file on github)

- Ring Flash-Bracket Connector (stl file on github)

- Arduino UNO R4

- LED with 220Ω resistor

  

  <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/CoverImage.png" alt="Flowchart" style="zoom: 33%;" />

## Software

- Visual Studio Code 1.100.2

- OpenCV-Python 4.11.0.86

- CUDA 11.8

- Pillow 10.4.0

- Python 3.8

- Pytorch 2.1

- Pyserial 3.5

- PySpin 1.1.1

- Torchvision 0.16.2

- Numpy 2.0.2

- Tensorflow 2.19.0

- Keras 3.9.2

- keyboard 0.13.5



## Dataset

  This project uses the **Metal nut class** dataset. The dataset contains high-resolution metal nuts.

The metal nuts class organized into three main directories:

- **`train/good/`**: Contains only normal metal nuts images without any defects. These images are used exclusively for training the autoenconder model in an unsupervised manner, allowing it to learn the statistical characteristics of normal metal nuts images.
- **`test/`**: Consists of both normal and anomalous samples. The sub-folders includes:
  - **`good/`**: normal metal nuts samples.
  - **`anomaly`**: various categories of metal nuts defects, such as surface manipulation of scratches at different parts of the metal nuts.

Each image is high-resolution and captured under controlled lighting conditions to reflect realistic industrial inspection settings. 

The data set could be downloaded after signing up on the linked site.

**Dataset link:** https://drive.google.com/drive/folders/1bW59Bby-a0kC1mEzdP6yaptvxSJK-ckU?usp=drive_link



------

# Method

## Preprocessing

  The process begins by detecting the largest contour and calculating its circumcenter. A fixed-size rectangle (400×400) is then drawn centered on this point, and the image is cropped accordingly. The cropped region is finally converted into a binary image for further analysis.

![](https://i.imgur.com/zad2XuA.png)



## Deep Learning Model

  A **Convolutional Autoencoder (CAE)** is a specialized type of autoencoder designed to process image data by leveraging convolution layers. Unlike traditional fully connected autoencoders, CAEs preserve the spatial structure of the input through the use of convolution and pooling operations. This makes them highly effective for tasks such as image denoising, anomaly detection, and compression.

- **Encoder:** This component applies a series of convolution and pooling layers to extract hierarchical features while progressively reducing the spatial dimensions of the input image. The output is a compact feature representation (**latent space**).
- **Decoder:** The decoder reconstructs the input image fro the latent representation by using transposed convolution layers, aiming to restore the spatial details of the original image.

<img src="https://i.imgur.com/J2DSIlM.png" style="zoom:80%;" />

![](https://i.imgur.com/beqLjX4.png)



## Evaluation

**Reconstruction Quality Evaluation**

To measure how well the model reconstructs input images, we used two quantitative metrics:

- **Mean Squared Error (MSE)**:
  This calculates the average of the squared differences between the original and reconstructed images. A lower MSE indicates higher reconstruction accuracy.
- **Structural Similarity Index Measure (SSIM)**:
  Unlike MSE, SSIM considers perceptual aspects of image similarity, making it more aligned with human visual perception. We used the `piq` library to compute SSIM and defined the loss as `1 - SSIM` to fit the optimization objective.



**Anomaly Detection Performance**

  Since the Autoencoder was used for unsupervised anomaly detection, we assessed its classification performance using reconstruction error thresholding. If the reconstruction error exceeded a predefined threshold, the input was classified as an anomaly. The following metrics were used to evaluate detection accuracy:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

These metrics were computed by comparing the predicted labels (based on the thresholded errors) with the ground truth labels provided in the test set.



**Error Distribution Visualization**

To provide a qualitative understanding of how well the model distinguishes between normal and abnormal data, we plotted the distribution of reconstruction errors for both classes.

- Normal and anomaly samples were separated based on ground truth labels.
- A histogram was generated for each group, with vertical dashed lines indicating the mean error for each class.
- This visualization clearly shows the difference in error distributions, supporting the effectiveness of reconstruction error as an anomaly indicator.



------

# Procedure

### 1. Model Training Flow

   The`model_training.py` file is on the github.

![](https://i.imgur.com/fNwKA0v.jpeg)

  Prior to deployment, a convolutional autoecoder is designed and trained using the metal nut class from the normal dataset. Only normal images are used to reflect the unsupervised learning setting. The dataset is preprocessed through resizing and normalization to fit the model's input format.

  The autoencoder consists of an encoder that compresses input images and a decoder that reconstructs them. The model learns to minimize reconstruction error on normal images. After training, the model is evaluated on both normal and defective test samples to determine a threshold value for anomaly detection. Once validated, the model is prepared for integration into the real-time inspection system. 

- **Load Data**

    A Convolutional Autoencoder is implemented to reconstruct normal images. The encoder compresses the input through convolution, ReLU activation, and max pooling, while the decoder restores the original size using upsampling and convolution layers. The final output is bounded between 0 and 1 using a Sigmoid function. Anomaly detection is performed based on the reconstruction error.

  ```python
  transform_train = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.RandomRotation(10),
      transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
  ])
  
  transform_test = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
  ])
  
  train_dataset = torchvision.datasets.ImageFolder(root='./NewData/train', transform=transform_train)
  train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True)
  
  test_dataset=torchvision.datasets.ImageFolder(root='./NewData/test',transform=transform_test)
  test_dataset.class_to_idx = {'good': 0, 'scratch':1}  # 원하는 매핑으로 재정의
  test_loader=DataLoader(test_dataset,batch_size=4, shuffle=False)
  test_dataset.samples = [
      (path, test_dataset.class_to_idx[os.path.basename(os.path.dirname(path))])
      for path, _ in test_dataset.samples
  ]
  ```

  

- **Convolution Autoencoder**

    The model consists of an encoder-decoder architecture designed to reconstruct input images. The encoder reduces spatial dimensions while capturing essential features using convolutional layers, ReLU activations, and max pooling. The decoder restores the original resolution through a series of upsampling and convolutional layers. A final Sigmoid activation ensures output pixel values remain between 0 and 1. This structure enables the detection of anomalies by comparing the reconstructed image to the original input.

  ```python
  class ConvAutoencoder(nn.Module):
      def __init__(self):
          super(ConvAutoencoder, self).__init__()
          self.encoder = nn.Sequential(
              nn.Conv2d(3, 64, 3, padding=1),   # [B, 64, 256, 256]
              nn.ReLU(),
              nn.MaxPool2d(2, 2, padding=0),    # [B, 64, 128, 128]
              nn.Conv2d(64, 128, 3, padding=1), # [B, 128, 128, 128]
              nn.ReLU(),
              nn.MaxPool2d(2, 2, padding=0),    # [B, 128, 64, 64]
              nn.Conv2d(128, 256, 3, padding=1),# [B, 256, 64, 64]
              nn.ReLU(),
              nn.MaxPool2d(2, 2, padding=0)     # [B, 256, 32, 32]
          )
          # Decoder
          self.decoder = nn.Sequential(
              nn.Conv2d(256, 256, 3, padding=1),# [B, 256, 32, 32]
              nn.ReLU(),
              nn.Upsample(scale_factor=2),      # [B, 256, 64, 64]
              nn.Conv2d(256, 128, 3, padding=1),# [B, 128, 64, 64]
              nn.ReLU(),
              nn.Upsample(scale_factor=2),      # [B, 128, 128, 128]
              nn.Conv2d(128, 64, 3, padding=1), # [B, 64, 128, 128]
              nn.ReLU(),
              nn.Upsample(scale_factor=2),      # [B, 64, 256, 256]
              nn.Conv2d(64, 3, 3, padding=1),   # [B, 3, 256, 256]
              nn.Sigmoid()
          )
  
  
      def forward(self, x):
          x = self.encoder(x)
          x = self.decoder(x)
          return x
  ```

  

- **Loss Function**

    A custom loss function combining Mean Squared Error (MSE) and Structural Similarity Index (SSIM) is implemented to better reflect perceptual quality. The loss is computed as a weighted sum of MSE and 1–SSIM, where the balance is controlled by a hyperparameter α. This hybrid formulation enhances reconstruction fidelity by considering both pixel-wise differences and structural information.

  ```python
  class HybridLoss(nn.Module):
    def __init__(self,alpha=0.5):
      super(HybridLoss,self).__init__()
      self.alpha=alpha
      self.mse=nn.MSELoss()
  
    def forward(self,recon,origin):
  
      ssim_loss=1-piq.ssim(recon,origin,data_range=1.0)
      mse_loss=self.mse(recon, origin)
  
      return self.alpha*mse_loss+(1-self.alpha)*ssim_loss
  ```

  

- **Model Training**

    The model is trained using the Adam optimizer with a learning rate of 1e-3 for 60 epochs. In each epoch, the training loop performs forward propagation to compute the reconstruction loss, followed by backpropagation and parameter updates. The average training loss per epoch is recorded to monitor convergence. 

  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  
  epochs=60
  
  for epochs in range(epochs):
    model.train()
    running_loss=0.0
    for batch in train_loader:
      inputs,_=batch
      inputs=inputs.to(device)
  
      #forward
      outputs=model(inputs)
      loss=loss_fn(outputs,inputs)
  
      #back propagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
      running_loss+=loss.item()
  
    avg_loss=running_loss/len(train_loader)
    print(f"Epoch [{epochs+1}/{epochs}] Loss: {avg_loss:.4f}")
  ```



- **Model Testing**

    Model performance is evaluated by computing the reconstruction error for each sample in the test set. The Mean Squared Error (MSE) between the input and the reconstructed output is used as the anomaly score. Samples with errors exceeding a predefined threshold are classified as anomalies. Accuracy is calculated by comparing the predicted labels with the ground truth. The function returns the overall accuracy along with the reconstruction errors and corresponding labels for further analysis. Set up the test dataset rate as 5:2 (good:anomaly).

  ```python
  def evaluate_accuracy(model, dataloader, loss_fn, threshold=0.01):
  
      model.eval()
      all_errors = []
      all_labels = []
      correct = 0
      total = 0
      total = len(dataloader.dataset)
  
      with torch.no_grad():
  
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
  
            # 재구성 에러 (MSE)
            reconstruction_error = F.mse_loss(outputs, inputs, reduction='none')
            reconstruction_error = reconstruction_error.mean(dim=[1,2,3])  # batch 단위 평균
            all_errors.extend(reconstruction_error.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
  
            # 예측: 오류가 threshold보다 크면 anomaly (1), 작으면 normal (0)
            preds = (reconstruction_error > threshold).float()
            
            # normal=0, anomaly=1 라벨로 구성되어 있음
            correct += (preds == labels).sum().item()
  
  
      acc = correct / total
      print(f'✅ Test Accuracy: {acc * 100:.2f}% (Threshold: {threshold})\n')
  
      return acc, all_errors, all_labels
  ```



- **Find Thresholding**

    To analyze the distribution of reconstruction errors, a histogram is plotted separately for normal and anomalous samples. The plot displays the frequency of reconstruction errors, with dashed vertical lines indicating the mean error for each class. This visualization helps distinguish normal and anomalous patterns based on error magnitude and supports the selection of an appropriate threshold.

  ```python
  def plot_recon_error_histogram_from_evaluation(errors, labels):
      normal_errors = [e for e, l in zip(errors, labels) if l == 0]
      anomaly_errors = [e for e, l in zip(errors, labels) if l == 1]
  
      plt.figure(figsize=(8, 5))
      plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', color='blue')
      plt.hist(anomaly_errors, bins=50, alpha=0.6, label='Anomaly', color='red')
      plt.axvline(np.mean(normal_errors), color='blue', linestyle='dashed', label='Normal Mean')
      plt.axvline(np.mean(anomaly_errors), color='red', linestyle='dashed', label='Anomaly Mean')
      plt.title("Reconstruction Error Histogram")
      plt.xlabel("Reconstruction Error")
      plt.ylabel("Count")
      plt.legend()
      plt.grid(True)
      plt.show()
  
  print("Testing for All test dataset")
  acc, all_errors, all_labels = evaluate_accuracy(model, test_loader, loss_fn, threshold=0.00066)
  plot_recon_error_histogram_from_evaluation(all_errors, all_labels)
  ```

  | ![](https://i.imgur.com/2Hzx2cP.png) |
  | :----------------------------------: |



- **Evaluation**

    To evaluate classification performance, the predicted labels are compared with the ground truth to compute the confusion matrix components: True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN). These values are visualized using a custom confusion matrix plot, where each quadrant is color-coded and annotated for clarity.

  Based on the confusion counts, common evaluation metrics are calculated, including Accuracy, Precision, Recall, and F1-score. These metrics provide a comprehensive assessment of the model's ability to detect anomalies while minimizing false alarms and missed detections.

  ```python
  def compute_confusion_counts(preds, labels):
      preds = np.array(preds).astype(int)
      labels = np.array(labels).astype(int)
  
      TP = np.sum((preds == 1) & (labels == 1))
      TN = np.sum((preds == 0) & (labels == 0))
      FP = np.sum((preds == 1) & (labels == 0))
      FN = np.sum((preds == 0) & (labels == 1))
  
      return TP, FN, FP, TN
  
  from matplotlib.patches import Rectangle
  
  def plot_confusion_map(TP, FN, FP, TN):
      fig, ax = plt.subplots(figsize=(6, 4))
  
      # 사각형 배경
      ax.add_patch(Rectangle((0, 1), 1, 1, facecolor='mediumseagreen'))  # TP
      ax.add_patch(Rectangle((1, 1), 1, 1, facecolor='cornflowerblue'))  # FN
      ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='cornflowerblue'))  # FP
      ax.add_patch(Rectangle((1, 0), 1, 1, facecolor='mediumseagreen'))  # TN
  
      # 텍스트: 위치, 내용, 색상
      ax.text(0.5, 1.5, f'True Positives\nTP = {TP}', ha='center', va='center', fontsize=12, color='white')
      ax.text(1.5, 1.5, f'False Negatives\nFN = {FN}', ha='center', va='center', fontsize=12, color='white')
      ax.text(0.5, 0.5, f'False Positives\nFP = {FP}', ha='center', va='center', fontsize=12, color='white')
      ax.text(1.5, 0.5, f'True Negatives\nTN = {TN}', ha='center', va='center', fontsize=12, color='white')
  
      # 라벨
      ax.text(-0.3, 1, 'Actual', va='center', ha='center', rotation='vertical', fontsize=13)
      ax.text(1, 2.05, 'Predicted', ha='center', fontsize=13)
  
      # 선, 눈금 제거
      ax.set_xlim(0, 2)
      ax.set_ylim(0, 2)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_aspect('equal')
      ax.axis('off')
  
      plt.title("Confusion Matrix", fontsize=14, pad=20)
      plt.tight_layout()
      plt.show()
      
  TP, FN, FP, TN = compute_confusion_counts(preds, labels)
  plot_confusion_map(TP, FN, FP, TN)
  
  Accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
  precision = TP / (TP + FP + 1e-6)
  recall = TP / (TP + FN + 1e-6)
  f1 = (2 * precision * recall) / (precision + recall + 1e-6)
  
  print(f"Accuracy: {Accuracy:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall:    {recall:.3f}")
  print(f"F1-score:  {f1:.3f}")
  ```



### 2. Camera Object Detection Flow

​	This algorithm is included in the `Anomaly_Detection.py`  which is on the github. This algorithm is for detecting the object when it comes to the frame of the camera. 



![Flowchart](https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/ObjectDetection_Flowchart.png)

- **`main()` Function**

    This report presents the `main()` function, which begins by verifying write permissions in its working directory through the creation and deletion of a temporary file, then acquires the FLIR Spinnaker system singleton and logs its SDK version before enumerating connected cameras and aborting with an informative message if none are found; for each detected camera it invokes `run_single_camera()`, accumulating success flags, and upon completion it explicitly deletes camera references, clears the camera list, releases the Spinnaker system instance, prompts the user to press Enter to acknowledge completion, and finally exits with code 0 on overall success or 1 on failure, thereby ensuring robust environment validation, device management, per‐camera execution, and clean resource teardown.

  ```python
  def main():
  
      try:
          # Attempt to create and open a temporary test file to verify write permissions
          test_file = open('test.txt', 'w+')
      except IOError:
          # If file creation fails, inform the user and exit the program gracefully
          print('Unable to write to current directory. Please check permissions.')
          input('Press Enter to exit...')
          return False
  
      # Close the test file now that write permissions are confirmed
      test_file.close()
  
      # Remove the temporary test file to clean up
      os.remove(test_file.name)
  
  
      # Initialize the overall result flag as True (will be updated based on camera operations)
      result = True
  
      # Retrieve singleton reference to system object
      system = PySpin.System.GetInstance()
  
      # Get current library version
      version = system.GetLibraryVersion()
      print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
  
      # Retrieve list of cameras from the system
      cam_list = system.GetCameras()
      
      # Retrieve the number of cameras currently connected to the system
      num_cameras = cam_list.GetSize()
  
      print('Number of cameras detected: %d' % num_cameras)
      # Finish if there are no cameras
      if num_cameras == 0:
  
          # Clear camera list before releasing system
          cam_list.Clear()
  
          # Release system instance
          system.ReleaseInstance()
  
          print('Not enough cameras!')
          input('Done! Press Enter to exit...')
          return False
  
      # Run example on each camera
      for i, cam in enumerate(cam_list):
          result &= run_single_camera(cam)
  
      # Release reference to camera
      del cam
  
      # Clear camera list before releasing system
      cam_list.Clear()
  
      # Release system instance
      system.ReleaseInstance()
  
      input('Done! Press Enter to exit...')
      return result
  
  
  if __name__ == '__main__':
      if main():
          sys.exit(0)
      else:
          sys.exit(1)
  ```

  

- **`run_single_camera()` Function**

    The `run_single_camera(cam)` function initializes a single FLIR camera instance by first obtaining its device nodemap, calling `cam.Init()` to activate the camera, and retrieving the GenICam nodemap to configure settings; it specifically checks for the availability and readability of the RGB8 pixel‐format entry, applies it if supported (printing confirmation) or warns the user otherwise, then calls `acquire_and_display_images()` to perform live image acquisition and processing—combining its Boolean outcome into the `result` flag via a bitwise AND—before deinitializing the camera with `cam.DeInit()`, catching any `PySpin.SpinnakerException` to log an error and mark the result as failed, and finally returning `True` if all operations succeeded or `False` if any step encountered an error.

  ```python
  def run_single_camera(cam):
  
      try:
          result = True
  
          nodemap_tldevice = cam.GetTLDeviceNodeMap()
  
          # Initialize camera
          cam.Init()
  
          # Retrieve GenICam nodemap
          nodemap = cam.GetNodeMap()
          node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
          node_pixel_format_rgb = node_pixel_format.GetEntryByName('RGB8')
  
          # Check if the RGB8 pixel format entry is available and readable
          if PySpin.IsAvailable(node_pixel_format_rgb) and PySpin.IsReadable(node_pixel_format_rgb):
              
              # Get the integer value corresponding to the RGB8 format
              pixel_format_rgb_value = node_pixel_format_rgb.GetValue()
              
              # Set the camera's pixel format to RGB8 using the retrieved value
              node_pixel_format.SetIntValue(pixel_format_rgb_value)
              print("PixelFormat set to RGB8.")
          else:
              # If RGB8 is not supported or not accessible, notify the user
              print("Unable to set PixelFormat to RGB8.")
  
          # Acquire images
          result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)
  
          # Deinitialize camera
          cam.DeInit()
  
      except PySpin.SpinnakerException as ex:
          print('Error: %s' % ex)
          result = False
  
      return result
  
  ```



- **`handle_close()` Function**

    The `handle_close(evt)` function serves as a Matplotlib figure close‐event callback that sets the global `continue_recording` flag to `False`, signaling the main acquisition loop to exit gracefully when the user closes the display window.

  ```python
  #Safe shutdown mechanism to terminate the camera capture loop.
  def handle_close(evt):
      global continue_recording
      continue_recording = False
  ```

​	

- **`acquire_and_display_images()` Function**

  T  his function performs real-time video streaming from a FLIR Spinnaker camera, detecting pixel changes between consecutive frames to identify anomalies and saving images at two-second intervals when significant motion is detected. It first sets the stream buffer handling mode to “NewestOnly” and switches to continuous acquisition, then captures an initial background frame for reference. Inside the acquisition loop, each new frame is grabbed, converted to RGB, cropped to the region of interest, and compared in grayscale to the previous frame to calculate a change ratio; if this ratio exceeds 4.95% -It was most appropriate value which is obtained through several experiment- and at least two seconds have passed since the last save, the frame is written to disk. The current frame is displayed live via Matplotlib, and if the Enter key is pressed, the loop ends, resources are cleanly released, and the function returns True (success) or False (failure).

  ```python
  def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
      global continue_recording
  
      result=True
      sNodemap = cam.GetTLStreamNodeMap()
  
      # Change bufferhandling mode to NewestOnly
      node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
      if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
          print('Unable to set stream buffer handling mode.. Aborting...')
          return False
  
      # Retrieve entry node from enumeration node
      node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
      if not PySpin.IsReadable(node_newestonly):
          print('Unable to set stream buffer handling mode.. Aborting...')
          return False
  
      # Retrieve integer value from entry node
      node_newestonly_mode = node_newestonly.GetValue()
  
      # Set integer value from entry node as new value of enumeration node
      node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
  
      print('*** IMAGE ACQUISITION ***\n')
      try:
          node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
          if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
              #print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
              return False
  
          # Retrieve entry node from enumeration node
          node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
          if not PySpin.IsReadable(node_acquisition_mode_continuous):
              #print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
              return False
  
          # Retrieve integer value from entry node
          acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
  
          # Set integer value from entry node as new value of enumeration node
          node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
          print('Acquisition mode set to continuous...')
  
          #Aquiring image
          cam.BeginAcquisition()
          print('Acquiring images...')
  
  
          device_serial_number = ''
          node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
          if PySpin.IsReadable(node_device_serial_number):
              device_serial_number = node_device_serial_number.GetValue()
              print('Device serial number retrieved as %s...' % device_serial_number)
  
          # Close program
          print('Press enter to close the program..')
  
          # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
          fig = plt.figure(1)
  
          # Close the GUI when close event happens
          fig.canvas.mpl_connect('close_event', handle_close)
  
          #create background image
          if cam.IsStreaming():
              backgournd_result = cam.GetNextImage(1000)
  
          else:
              print("Camera is not streaming.")
              return False
          
          
          #Save initial background 
          raw_background=backgournd_result
          processor = PySpin.ImageProcessor()
          processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
          background_rgb=processor.Convert(raw_background, PySpin.PixelFormat_RGB8)
          background=background_rgb.GetNDArray()
          background=background[start_y:start_y+test_width,start_x:start_x+test_width]
          
          
          #Initialize time of precious frame
          last_saved_time=0
          
          #Initialize the flag for judging whether the nut is anomaly 
       
         
          #Initialize previous frame
          before=np.zeros_like(background)
          
          # Retrieve and display images
          while(continue_recording):
              
              try:
                  
                  #Acquire the image from the camera.
                  if cam.IsStreaming():
                      image_result = cam.GetNextImage(1000)
  
                  else:
                      print("Camera is not streaming.")
                      return False
                  
  
                  #  Ensure image completion
                  if image_result.IsIncomplete():
                      print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
  
                  else:                    
                  
                      #current: 700x700 raw image
                      raw_image=image_result
                      processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
                      image_rgb = processor.Convert(raw_image, PySpin.PixelFormat_RGB8)
                      current = image_rgb.GetNDArray().copy()
                      current=current[start_y:start_y+test_width,start_x:start_x+test_width]
                      
                      #Gray images for calculating difference between previous and current frame
                      gray_current=cv.cvtColor(current,cv.COLOR_RGB2GRAY)
                      gray_before=cv.cvtColor(before,cv.COLOR_RGB2GRAY)
  
                      #Calculate difference between previous and current frame
                      diff=cv.absdiff(gray_current,gray_before)
                      _,diff=cv.threshold(diff,100,255,cv.THRESH_BINARY)
                      
                      #Declare change ratio 
                      change_ratio = np.sum(diff > 0) / diff.size
                      
  
                      #Time of current frame
                      current_time=time.time()
                      
                      #Initialize variable for judge whether picture is taken
                      picture_is_taken=0
                      
                      
                      #If Change ratio is bigger than 4.95% and more than two seconds have passed since the last photo was taken, save the frame
                      if (change_ratio > 0.0495) and ((current_time-last_saved_time)>2) :  
                          
                          last_saved_time = current_time
                          
                          
                          timestamp = time.strftime("%Y%m%d_%H%M%S")  
                          
                          
                          os.makedirs(save_dir,exist_ok=True)
                          current_bgr=cv.cvtColor(current, cv.COLOR_RGB2BGR)
                          #cropped_before_bgr=cv.cvtColor(cropped_before, cv.COLOR_RGB2BGR)
  
                          filename = os.path.join(save_dir, f"detected_{timestamp}.jpg")
                          cv.imwrite(filename, current_bgr)
                          #print(f" Object detected! Image saved as {filename}")
                          
                          #Notification: Photo has been taken
                          picture_is_taken=1
                          
                         
                      #To detect other objects as well, save current frame as previous frame
                      before=current
                      
                      #Display the current image
                      plt.imshow(current)
                      plt.pause(0.001)
                      plt.clf()
                      
                      # If user presses enter, close the program
                      if keyboard.is_pressed('ENTER'):
                          print('Program is closing...')
                          
                          # Close figure
                          plt.close('all')             
                          input('Done! Press Enter to exit...')
                          continue_recording=False                        
  
                  image_result.Release()
  
              except PySpin.SpinnakerException as ex:
                  # Catch any Spinnaker-specific exceptions during image acquisition/processing
                  print('Error: %s' % ex)
                  # Return False to indicate that the acquisition loop failed
                  return False
  
          cam.EndAcquisition()
  
      except PySpin.SpinnakerException as ex:
          # Catch any Spinnaker-specific exceptions around the overall acquisition setup/teardown
          print('Error: %s' % ex)
  
          # Return False to signal that the main function encountered an error
          return False
  
      return True
  
  ```

  

  When display `diff` inside the function, it looks like this: 

  |                    Object is not detected                    |                      Object is detected                      |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![Flowchart](https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/NoObject.png) | ![Flowchart](https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/ObjectDetected.png) |

  The final Detected images are saved as follows:

  |                         A Normal Nut                         |                       An Abnormal Nut                        |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/DetectedImage_Normal.jpg" alt="Flowchart" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/DetectedImage_Abnormal.jpg" alt="Flowchart" style="zoom:50%;" /> |



###  3. Image Preprocessing

  The process is included in `Anomaly_Detection.py`

![Flowchart](https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Image_Preprocessing_Flowchart.png)



  To ensure consistency and enhance the quality of training data, all input images were processed through a series of steps. First, image files were loaded from the specified directory, and unsupported file formats were excluded. Each valid image was converted to grayscale, and a sharpening filter was applied to emphasize edge details.

  A binary threshold was then used to isolate the metal nut from the background. The largest contour was extracted, and its minimum enclosing circle was calculated to determine the object’s center. A square region of 400×400 pixels was cropped around this center point. Boundary checks were included to prevent cropping beyond the image dimensions.

  The cropped region was saved in grayscale format to the target directory. This preprocessing step helped standardize the image position, scale, and format, facilitating more stable and effective model training.

```python
 if picture_is_taken:
                        
    #source Image
    source=cv.imread(filename)
    source_gray=cv.cvtColor(source,cv.COLOR_RGB2GRAY)

    #Find threshold of the image
    _, thresh = cv.threshold(source_gray, 127, 255, cv.THRESH_BINARY)

    #Find the largest contour of the image
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)                       
    
    #Continue the process only when the objet contour is captured
    if contours: 
        largest_contour = max(contours, key=cv.contourArea)
    else:
        continue
    
    #Copy the source image and draw the largest contour on it
    contour_image=source.copy()
    cv.drawContours(contour_image,[largest_contour],-1,(255,0,0),3)
    
    #Find circumscribed circle and center point of the largest contour 
    (x, y), radius = cv.minEnclosingCircle(largest_contour)
    center_x, center_y = int(x), int(y)

    #Calculate crop area
    crop_size = 400
    half_crop = crop_size // 2
    h, w = source.shape[:2]

    #Exception handling to prevent exceeding image boundaries
    x1 = max(center_x - half_crop, 0)
    y1 = max(center_y - half_crop, 0)
    x2 = min(center_x + half_crop, w)
    y2 = min(center_y + half_crop, h)

    #Crop a 400×400 pixel region centered on the previously obtained contour’s center
    cropped_source=source[y1:y2, x1:x2]

    #Treat the cropped image as the image to send to the model.
    image_to_model=cv.cvtColor(cropped_source,cv.COLOR_RGB2GRAY)
    
    #Save the cropped image in your directory
    cropped_filename = os.path.join(save_dir, f"cropped_detected_{timestamp}.jpg")
    cv.imwrite(cropped_filename, image_to_model)

    #Convert the preprocessed image into a tensor format suitable for input to the model.
    taken_nut = Image.open(cropped_filename).convert("RGB")
    input_tensor = transform_test(taken_nut)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
```

​		

### 4. Model Prediction Flow

`predict_anomaly()` is located in `Anomaly_Detection.py`.

<img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Model_Prediction_Flowchart.png"  />

This function is designed to predict whether an input image tensor is anomalous using the provided model. It begins by switching the model to evaluation mode (`model.eval()`), then, inside a `torch.no_grad()` block, performs a forward pass to obtain the reconstructed output. It calculates the pixel-wise mean squared error (MSE) between the original and reconstructed images, averaging over the spatial and channel dimensions for each batch item. If this MSE exceeds the threshold set during model training, it assigns the label `'a1'` (Anomaly); otherwise, it assigns `'a0'` (Normal). The function prints the reconstruction error to six decimal places and displays the prediction with an emoji on the console, then returns the string `'a0'` or `'a1'`, which is sent to the Arduino as a flag.

```python
def predict_anomaly(model, image_tensor, threshold):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        mse = F.mse_loss(output, image_tensor, reduction='none')
        mse = mse.mean(dim=[1, 2, 3]) 
        pred='a0'

        #Anomaly
        if mse>threshold:
            pred='a1'

        print(f"Reconstruction Error: {mse.item():.6f}")
        print("Prediction:", "🤬 Anomaly" if pred == 'a1' else "😁 Normal")
        
        return pred
```



### 5. Real-Time Inference Flow

![](https://i.imgur.com/60hZx3c.jpeg)

  In deployment, metal nuts are carried on a conveyor belt and imaged by a Guppy F-080 camera. Each image is preprocessed and passed through the trained autoencoder. The system calculates the pixel-wise reconstruction error between the original and reconstructed images. If the computed error exceeds a predefined threshold, the metal nut is classified as defective. In such cases, a red LED is triggered as a visual alert, enabling immediate identification and response to the anomaly. It the error remains below the threshold, the metal nut is considered normal and continues along the production line without intervention.



## Tutorials

### Hardware Setting

#### Camera Setting

 The following hardware was used in the project:

|                    Machine Vision Camera                     |                          Ring Flash                          |                 **Camera-Bracket Connector**                 |                 Ring Flash-Bracket Connector                 |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Camera.jpg" alt="Flowchart" style="zoom: 33%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/RingFlash.jpg" alt="Flowchart" style="zoom: 33%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Camera_Bracket_Connector.jpg" alt="Flowchart" style="zoom: 33%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/RingFlash_Bracket_Connector.jpg" alt="Flowchart" style="zoom: 33%;" /> |
|                       **Camera Cable**                       |                      **Metal  Profile**                      |                      **Conveyor Belt**                       |                                                              |
| <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/CameraCable.jpg" alt="Flowchart" style="zoom: 33%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Bracket.jpg" alt="Flowchart" style="zoom: 33%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/ConveyerBelt.jpg" alt="Flowchart" style="zoom: 25%;" /> |                                                              |



The complete system assembled from these components is as follows:	

| Front View                                                   | Top View                                                     | Rear View                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Front.jpg" alt="Flowchart" style="zoom: 33%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Up.jpg" alt="Flowchart" style="zoom: 33%;" /> | <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Back.jpg" alt="Flowchart" style="zoom: 33%;" /> |

Then, set the light as 10.



The height from the ground to the light, and the distance from the bracket to the light are as follows:

| <img src="https://raw.githubusercontent.com/YeChanKimm/DLIP/main/FinalProject_ReportImages/Back_with_text.jpg" alt="Flowchart" style="zoom: 25%;" /> |
| ------------------------------------------------------------ |

- a=160mm
- b=184mm



#### Conveyor Belt Setting

![Image](https://github.com/user-attachments/assets/4bf9d881-d9e2-4563-b6e3-82da6032de5d)

- Follow the picture to connect arduino board with step motor driver.

- Set applied voltage on power supply over 20[V], unless the motor will not work.

  

### Installation

#### 1) Serial Communication

  The `pyserial` library is required to enable serial communication between the Python script and the Arduino board. It allows the Python program to send and receive data through the computer's serial (COM) port.

```bash
pip install pyserial
```

**Instruction with sample code:** [Serial Communication](https://github.com/HaeunKim2/DLIP_FINAL/blob/main/Serial%20Communication%20Between%20Python%20and%20Arduino.md)



#### 2) Machine Vision Camera

  To use Machine Vision Camera, PySpin module is necessary. Used a Python version (3.8) compatible with the PySpin module to ensure proper integration.

- Create New environment

  ```
  conda create -n py38 python=3.8
  ```

- Neccesary module Download

  ```
  pip install matplotlib
  pip install keyboard
  pip install PySpin
  ```

- Necessary Python package설치

  ```
  conda activate envname
  cd .whl파일 존재 경로
  python -m pip install *filename*.whl
  ```

**Instruction:**  [Machine Vision Camera](https://github.com/HaeunKim2/DLIP_FINAL/blob/d930311b1175893cc912941333e69ee92c0e6724/Spinnaker%20SDK%20%26%20Python%20%EC%97%B0%EB%8F%99.md)



#### 3) Image Quality Evaluation

   The `piq` library is a PyTorch-based tool used to measure image quality. In this project, we installed it to better evaluate how well the Autoencoder model reconstructs images. Instead of using only Mean Squared Error (MSE), which doesn't fully reflect visual quality, we used SSIM (Structural Similarity Index Measure) from `piq` to calculate loss in a way that matches human perception more closely.

```bash
pip install piq
```



### Program Implementation

  Download `conveyor_belt.ino`  from github or copy and paste the code below to the Arduino IDE. Then upload the code on the Arduino UNO R4 before run the python code. If the python code run ahead, uploading arduino code on the board is not available because the port is already busy.

```c
const int pulPin = 3;
const int dirPin = 4;
const int enPin  = 5;
const int ledPin = 6;
volatile bool stopFlag = false;
int x;

int stepCount = 0;
unsigned long lastStepTime = 0;
const int totalSteps = 400 * 5;
const int stepDelay = 12000;  // in microseconds

void setup() {
  pinMode(pulPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode( enPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  digitalWrite(dirPin, LOW);  // 정방향
  digitalWrite( enPin, LOW);    // 드라이버 활성화
  digitalWrite(ledPin, LOW);

  Serial.begin(9600);
}

void loop() {
  
  // 파이썬 코드에서 flag 받아오기 
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {                 // flag = 1 이 들어오면 불량
      stopFlag = true;                // 모터 stop flag
      digitalWrite( enPin, HIGH);     // 모터 드라이버 비활성화 (모터 정지)
      digitalWrite(ledPin, HIGH);     // LED 켜기
      stepCount = 0;  // 현재 회전 중단

    } else if (cmd == '0') {          // flag = 0 이 들어오면 정상
      stopFlag = false;               // 모터 계속 작동
      digitalWrite( enPin, LOW);      // 모터 드라이버 활성화 (모터 동작)
      digitalWrite(ledPin, LOW);      // LED 끄기

      stepCount = 0;                  // 새 회전 시작
      lastStepTime = micros();
    }
  }

  //모터 회전 처리 (non-blocking)
  if (!stopFlag) {
    unsigned long now = micros();
    if (now - lastStepTime >= stepDelay * 2) {
      digitalWrite(pulPin, HIGH);
      delayMicroseconds(stepDelay);
      digitalWrite(pulPin, LOW);
      lastStepTime = now;
      stepCount++;
      if (stepCount >= totalSteps) stepCount = 0;
    }
  }
}
```



  Download `CAEmodel.pth`  from github. Then also download `Anomaly_Detection.py` or copy and paste the code below to visual studio. After connecting camera with laptop, run this code. If not, "not enough cameras. Number of cameras detected: 0" alert will be printed. 

```python
#========================PySpin=============================
import os
import pyspin as PySpin
import matplotlib.pyplot as plt
import sys
import keyboard
import time
import serial
#========================Model============================
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
#=========================OpenCV=============================
import numpy as np
import cv2 as cv

#Arduino setting
ser = serial.Serial('COM6', 9600)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
current_cwd = os.getcwd()
print(f"Present working directory: {current_cwd}")

#Create Directory which Detected image will be saved as you want
save_dir='Detected_images'

#Setting device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
global continue_recording
continue_recording = True

#ROI parameters setting for fixed location of the camera 
start_x=162
start_y=290
test_width=700

#dataset transformer
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

text=0

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#Function for judging whether a nut is defected
def predict_anomaly(model, image_tensor, threshold):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        mse = F.mse_loss(output, image_tensor, reduction='none')
        mse = mse.mean(dim=[1, 2, 3]) 
        pred='a0'

        #Anomaly
        if mse>threshold:
            pred='a1'

        print(f"Reconstruction Error: {mse.item():.6f}")
        print("Prediction:", "🤬 Anomaly" if pred == 'a1' else "😁 Normal")
        
        return pred

#Safe shutdown mechanism to terminate the camera capture loop.
def handle_close(evt):
    global continue_recording
    continue_recording = False

#Acquiring and displaying image
def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    global continue_recording

    result=True
    sNodemap = cam.GetTLStreamNodeMap()

    # Change bufferhandling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    print('*** IMAGE ACQUISITION ***\n')
    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            #print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            #print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        print('Acquisition mode set to continuous...')

        #Aquiring image
        cam.BeginAcquisition()
        print('Acquiring images...')

        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Close program
        print('Press enter to close the program..')

        # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
        fig = plt.figure(1)

        # Close the GUI when close event happens
        fig.canvas.mpl_connect('close_event', handle_close)

        # Setting GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model instantiation and loading pre-trained weights
        model = ConvAutoencoder().to(device)
        model.load_state_dict(torch.load("FINAL_MODEL.pth", map_location=device))
        model.eval()  

        #create background image
        if cam.IsStreaming():
            backgournd_result = cam.GetNextImage(1000)

        else:
            print("Camera is not streaming.")
            return False
        
        #Save initial background 
        raw_background=backgournd_result
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        background_rgb=processor.Convert(raw_background, PySpin.PixelFormat_RGB8)
        background=background_rgb.GetNDArray()
        background=background[start_y:start_y+test_width,start_x:start_x+test_width]
        
        #Initialize time of precious frame
        last_saved_time=0
        
        #Initialize the flag for judging whether the nut is anomaly 
        flag = 'a0'
        ser.write(str(flag).encode()) 
       
        #Initialize previous frame
        before=np.zeros_like(background)
        
        # Retrieve and display images
        while(continue_recording):
            
            try: 
                #Acquire the image from the camera.
                if cam.IsStreaming():
                    image_result = cam.GetNextImage(1000)
                else:
                    print("Camera is not streaming.")
                    return False

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:                     
                    # Getting the image data as a numpy array
                    flag = 'a0'
                    ser.write(str(flag).encode()) 
                    
                    #current: 700x700 raw image
                    raw_image=image_result
                    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
                    image_rgb = processor.Convert(raw_image, PySpin.PixelFormat_RGB8)
                    current = image_rgb.GetNDArray().copy()
                    current=current[start_y:start_y+test_width,start_x:start_x+test_width]
                    
                    #Gray images for calculating difference between previous and current frame
                    gray_current=cv.cvtColor(current,cv.COLOR_RGB2GRAY)
                    gray_before=cv.cvtColor(before,cv.COLOR_RGB2GRAY)

                    #Calculate difference between previous and current frame
                    diff=cv.absdiff(gray_current,gray_before)
                    _,diff=cv.threshold(diff,100,255,cv.THRESH_BINARY)
                    
                    #Declare change ratio 
                    change_ratio = np.sum(diff > 0) / diff.size

                    #Time of current frame
                    current_time=time.time()
                    
                    #Initialize variable for judge whether picture is taken
                    picture_is_taken=0
                    
                    #If Change ratio is bigger than 4.95% and more than two seconds have passed since the last photo was taken, save the frame
                    if (change_ratio > 0.0495) and ((current_time-last_saved_time)>2) :  
                        
                        last_saved_time = current_time
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")  
                        
                        os.makedirs(save_dir,exist_ok=True)
                        current_bgr=cv.cvtColor(current, cv.COLOR_RGB2BGR)
                        #cropped_before_bgr=cv.cvtColor(cropped_before, cv.COLOR_RGB2BGR)

                        filename = os.path.join(save_dir, f"detected_{timestamp}.jpg")
                        cv.imwrite(filename, current_bgr)
                        #print(f" Object detected! Image saved as {filename}")
                        
                        #Notification: Photo has been taken
                        picture_is_taken=1  
                       
                    #To detect other objects as well, save current frame as previous frame
                    before=current

                    #Image prepocessing for taken image
                    if picture_is_taken:
                        
                        #source Image
                        source=cv.imread(filename)
                        source_gray=cv.cvtColor(source,cv.COLOR_RGB2GRAY)

                        #Find threshold of the image
                        _, thresh = cv.threshold(source_gray, 127, 255, cv.THRESH_BINARY)

                        #Find the largest contour of the image
                        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)                       
                        #Continue the process only when the objet contour is captured
                        if contours: 
                            largest_contour = max(contours, key=cv.contourArea)
                        else:
                            continue
                        
                        #Copy the source image and draw the largest contour on it
                        contour_image=source.copy()
                        cv.drawContours(contour_image,[largest_contour],-1,(255,0,0),3)
                        
                        #Find circumscribed circle and center point of the largest contour 
                        (x, y), radius = cv.minEnclosingCircle(largest_contour)
                        center_x, center_y = int(x), int(y)

                        #Calculate crop area
                        crop_size = 400
                        half_crop = crop_size // 2
                        h, w = source.shape[:2]

                        #Exception handling to prevent exceeding image boundaries
                        x1 = max(center_x - half_crop, 0)
                        y1 = max(center_y - half_crop, 0)
                        x2 = min(center_x + half_crop, w)
                        y2 = min(center_y + half_crop, h)

                        #Crop a 400×400 pixel region centered on the previously obtained contour’s center
                        cropped_source=source[y1:y2, x1:x2]

                        #Treat the cropped image as the image to send to the model.
                        image_to_model=cv.cvtColor(cropped_source,cv.COLOR_RGB2GRAY)
                        
                        #Save the cropped image in your directory
                        cropped_filename = os.path.join(save_dir, f"cropped_detected_{timestamp}.jpg")
                        cv.imwrite(cropped_filename, image_to_model)

                        #Convert the preprocessed image into a tensor format suitable for input to the model.
                        taken_nut = Image.open(cropped_filename).convert("RGB")
                        input_tensor = transform_test(taken_nut)
                        input_tensor = input_tensor.unsqueeze(0)
                        input_tensor = input_tensor.to(device)
                        
                        #Put the tensor in the Autoencoder model and set the suitable threshold as we got from model training process 
                        flag = predict_anomaly(model, input_tensor, threshold=0.0011)
                        
                        #Send the flag to the arduino. When an anomalous nut is detected, the Arduino turns on the LED and stops the conveyor belt.
                        ser.write(str(flag).encode()) 
                        
                        #When an anomalous nut is detected, wait until the Arduino has completed the commands to stop the conveyor belt and turn on the LED, and sends back ‘ready’.
                        if flag == 'a1':
                            while True:
                                if ser.in_waiting > 0:  
                                    line = ser.readline().decode().strip()
                                    
                                    if line == 'ready':
                                        break
                    
                    #Display the current image
                    plt.imshow(current)
                    plt.pause(0.001)
                    plt.clf()
                    
                    # If user presses enter, close the program
                    if keyboard.is_pressed('ENTER'):
                        print('Program is closing...')
                        
                        # Close figure
                        plt.close('all')             
                        input('Done! Press Enter to exit...')
                        continue_recording=False                        

                image_result.Release()

            except PySpin.SpinnakerException as ex:
                # Catch any Spinnaker-specific exceptions during image acquisition/processing
                print('Error: %s' % ex)
                # Return False to indicate that the acquisition loop failed
                return False

        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        # Catch any Spinnaker-specific exceptions around the overall acquisition setup/teardown
        print('Error: %s' % ex)

        # Return False to signal that the main function encountered an error
        return False

    return True


def run_single_camera(cam):

    try:
        result = True
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        node_pixel_format_rgb = node_pixel_format.GetEntryByName('RGB8')

        # Check if the RGB8 pixel format entry is available and readable
        if PySpin.IsAvailable(node_pixel_format_rgb) and PySpin.IsReadable(node_pixel_format_rgb):
            
            # Get the integer value corresponding to the RGB8 format
            pixel_format_rgb_value = node_pixel_format_rgb.GetValue()
            
            # Set the camera's pixel format to RGB8 using the retrieved value
            node_pixel_format.SetIntValue(pixel_format_rgb_value)
            print("PixelFormat set to RGB8.")
        else:
            # If RGB8 is not supported or not accessible, notify the user
            print("Unable to set PixelFormat to RGB8.")

        # Acquire images
        result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result

def main():
    try:
        # Attempt to create and open a temporary test file to verify write permissions
        test_file = open('test.txt', 'w+')
    except IOError:
        # If file creation fails, inform the user and exit the program gracefully
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    # Close the test file now that write permissions are confirmed
    test_file.close()

    # Remove the temporary test file to clean up
    os.remove(test_file.name)

    # Initialize the overall result flag as True (will be updated based on camera operations)
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    
    # Retrieve the number of cameras currently connected to the system
    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)
    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):
        result &= run_single_camera(cam)

    # Release reference to camera
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
```



------

# Results and Analysis

## 1. Final Result

  To validate the model's performance, the input image, reconstructed output, and reconstruction error map are visualized for both normal and anomalous samples. For the normal sample, the reconstruction closely matches the input, resulting in minimal error across the image. In contrast, the anomalous sample shows clear reconstruction discrepancies, especially around the defect regions, highlighted in red on the error map. These visual results demonstrate the model’s ability to distinguish defects based on pixel-wise reconstruction error.

| ![](https://i.imgur.com/lNfJIxI.png) |
| :----------------------------------: |
| ![](https://i.imgur.com/EaWTWE9.png) |

  The performance of the proposed autoencoder-based metal nut defect detection system was evaluated using a test dataset containing both normal and anomalous samples. Reconstruction error histograms indicate a clear separation between the error distributions of normal and defective samples. Normal samples showed relatively low reconstruction errors centered around 0.0010, while anomalies exhibited higher errors, mostly concentrated around 0.0012.

  Using a fixed threshold for classification, the confusion matrix shows that the model achieved 19 true positives (TP), 1 false negative (FN), 15 false positives (FP), and 37 true negatives (TN). Based on these counts, the overall accuracy was 77.8%, with a precision of 55.9%, recall of 95.0%, and an F1-score of 70.4%.

  These results suggest that the model is highly effective at identifying defective metal nuts (high recall), although it exhibits a moderate level of false alarms (lower precision). The high recall indicates the system’s suitability for safety-critical defect screening applications where missing a defect is more critical than a false alarm.

| ![](https://i.imgur.com/hcjSD3Y.png) | ![](https://i.imgur.com/mB7MzhP.png) |
| :----------------------------------: | :----------------------------------: |

| ![](https://i.imgur.com/03QQioT.png) | <img src="https://i.imgur.com/I1eJmrY.png" style="zoom:150%;" /> |
| :----------------------------------: | :----------------------------------------------------------: |



## 2. Discussion

  The primary goal of this project was to develop an anomaly detection system based on an Autoencoder trained solely on normal data, with the ability to effectively identify defective samples. The final model achieved an overall **accuracy of 77.8%**, thereby meeting the initial target of **above 75%**. Although the accuracy figure may seem relatively low at first glance, the more critical performance metric in this context was recall. In quality inspection systems, missing defective samples (false negatives) can lead to significant downstream issues. Therefore, capturing all defective items, even at the cost of some false positives, was prioritized. In this regard, the model achieved a **recall of 95.0%**, indicating a strong ability to detect defective cases. This result suggests that the model is suitable for deployment in real-world quality control pipelines, where missing a defect is far more costly than an occasional false alarm. Further improvements in precision and a more refined analysis of reconstruction errors may lead to a more balanced and robust system.

  In our experiments, the most critical factor was maintaining consistency between the controlled training environment and the live inspection setup. Without dedicated preprocessing, nuts on the moving conveyor belt often appeared in shifted positions or under different lighting conditions, causing the autoencoder—trained exclusively on static, centered, grayscale images—to misclassify normal nuts as anomalies. To address this, we applied classic image‐processing techniques (thresholding, contouring, filtering, and region‐of‐interest extraction) to isolate and crop each nut before converting it to grayscale. This pipeline guaranteed that every input matched the spatial and intensity characteristics of the training data.

  Equally important was the stabilization of the physical environment. We fixed the camera mount, calibrated the conveyor‐belt motor speed to 41.67 steps per second (SPS) to prevent motion blur, and fine-tuned illumination to level 10 on our CVT-Emini K-4Ch fixtures. These adjustments minimized background noise and emphasized surface defects, ensuring that the model’s anomaly scores reflected genuine irregularities rather than environmental artifacts. As a result, our system achieved a substantial accuracy gain—nearly 10 percentage points above the reference study.

  Given the safety-critical context of defect detection, we deliberately optimized for maximum recall, accepting a lower precision to guarantee that almost no defective nut goes undetected. Under a test distribution of five normal nuts for every two defective ones, this high-recall strategy yielded 95 % recall and 55 % precision. Because our performance metric is heavily influenced by the proportion of positive samples, we expect that increasing the ratio of defective nuts in real-world validation will further raise overall accuracy. Future work should investigate adaptive decision thresholds and cost-sensitive training to balance recall and precision dynamically, tailoring the system to varying defect rates and operational risk tolerances.



# Reference

Complete list of all references used (github, blog, paper, etc)

- https://github.com/ykkimhgu/MIP2024-SteelAnomalyDetection/tree/main
- [025. 합성곱 오토인코더 아싸 좋구나](https://zir2-nam.tistory.com/entry/025-합성곱-오토인코더-아싸-좋구나)
- Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD -- A comprehensive real-world dataset for unsupervised anomaly detection*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 9592–9600). https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf
- https://yeong-jin-data-blog.tistory.com/entry/Autoencoder



# Contribution

**Yechan Kim:** 

- Designed the basic structure of the convolutional autoencoder (CAE) model

- Trained the model using the prepared dataset

- Set up the system environment and connected the machine vision camera

- Implemented real-time camera feed display on the monitor

- Programmed object detection and image capture-storing algorithm

- Integrated each systems-Pre-trained model, Arduino, Object detecting algorithm- to construct the final system.

  

**Haeun Kim:** 

- Developed hardware components with 3D printing
- Assembled a mounting structure using aluminum profiles
- Implemented conveyor belt control via Arduino
- Integrated LED indicators for system feedback with Serial Communication
- Composed the project report
- Build dataset and Image preprocessing for dataset
- Debugged and developed the CAE model and Trained the model
- Determined the anomaly detection threshold based on reconstruction error analysis
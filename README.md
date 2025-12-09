# ECE 792 Final Project
# Sim-to-Cloud: Cloud-based AI Model Deployment on Isaac Sim-generated Synthetic Data 


---

## 📌 1. Motivation & Problem Statement

Modern computer vision (CV) systems depend heavily on high-quality labeled datasets. However, collecting and annotating large-scale image datasets in the real world is **time-consuming**, **expensive**, and often **infeasible** in scenarios involving rare events or controlled environments. Synthetic data has emerged as an effective alternative, allowing developers to generate customizable, scalable datasets that closely mimic real-world conditions.

This project aims to explore how **synthetically generated data** can be integrated into a **cloud-based AI deployment pipeline**. By using **NVIDIA Isaac Sim’s Replicator API**, thriugh which we render images. These assets are then exported to **AWS S3**, enabling secure, scalable storage. A **pre-trained model**, the YOLOv8 in our case, is hosted in the **AWS SageMaker** which is used solely for **inference**, allowing the pipeline to test how synthetic inputs perform in downstream CV applications.

It is essential because synthetic data generation unlocks opportunities for developing robust models while reducing dataset acquisition costs. Evaluating such data through a deployed cloud inference system helps validate the **feasibility and usefulness** of synthetic data in modern AI workflows.

---

## 📌 2. Use of Course Concepts

This project incorporates multiple concepts explored throughout the course, directly applying them to a real-world cloud AI workflow:

### **✔ Cloud Computing**
- **Amazon S3** for cloud object storage of synthetic images and annotations.
- **AWS SageMaker** for model hosting, inference, and managed compute resources.  

### **✔ Virtualization & Scalability**
- SageMaker endpoints automatically scale based on incoming inference requests.  
- S3 provides virtually unlimited storage for synthetic datasets.  

### **✔ Reproducibility & CI/CD**
- **GitHub Actions** is used to automate:
  - Deployment of SageMaker inference endpoints.  
  - Synchronization of data to S3.  
  - Code version tracking and testing.  
- Ensures full reproducibility of the deployment pipeline.  

### **✔ Monitoring & Logging**
- Inference outputs and execution logs are captured through SageMaker.

### **✔ Ethical Considerations**
- Synthetic data avoids privacy issues associated with real-world datasets.  
- Reduces reliance on human subjects and potential biases in manual annotations/  

---

## 📌 3. Documentation & Workflow Explanation

### **Overall Workflow**

![Sim-to-Cloud Flowchart Diagram](figs/sim2cloud.png)

The above attached diagram describes a three-stage pipeline designed to generate synthetic training data from real-world images and utilize it to deploy a cloud-based object detection model.

  - **Stage 1: Data Acquisition & Preprocessing**: The workflow begins with real-world, single-view images. These images are processed using SAM 3D       (Segment Anything Model 3D) to create segmented and 3D reconstructed meshes of the target objects. This data acquisition and preprocessing step is adapted from one of our earlier projects (https://github.com/dayyapp_ncstate/ECE875_SoGo.git) that also utilizes this SAM 3D model in an inference manner. 

  - **Stage 2: Synthetic Data Generation**: The resulting 3D meshes are first converted into the USD format and then imported into the NVIDIA Isaac Sim platform. Later, with the help of the Isaac Sim's Synthetic Data Recorder, diverse rendered products of the synthetic scenes are created outputting a dataset of synthetic 2D RGB images.

  - **Stage 3: Model Training & Inference**: The newly generated synthetic images are moved to the cloud and stored in AWS S3 Buckets. This data is then consumed by Amazon SageMaker to train and deploy a YOLOv8 (You Only Look Once) object detection model for inference tasks.

# AIMD-Sim-to-Cloud : Synthetic Data Pipeline for Computer Vision via AWS
*Course Project — Cloud-Based AI Deployment*

---

## 📌 1. Motivation & Problem Definition

Modern computer vision systems depend heavily on high-quality labeled datasets. However, collecting and annotating large-scale image datasets in the real world is **time-consuming**, **expensive**, and often **infeasible** in scenarios involving rare events or controlled environments. Synthetic data has emerged as an effective alternative, allowing developers to generate customizable, scalable datasets that closely mimic real-world conditions.

This project aims to explore how **synthetically generated data** can be integrated into a **cloud-based AI deployment pipeline**. By using **NVIDIA Isaac Sim’s Replicator API**, we generate images. These assets are then exported to **AWS S3**, enabling secure, scalable storage. A **pre-trained model** hosted in **AWS SageMaker** is used solely for **inference**, allowing the pipeline to test how synthetic inputs perform in downstream CV applications.

It is essential because synthetic data generation unlocks opportunities for developing robust models while reducing dataset acquisition costs. Evaluating such data through a deployed cloud inference system helps validate the **feasibility and usefulness** of synthetic data in modern AI workflows.

---

## 📌 2. Use of Course Concepts

This project incorporates multiple concepts explored throughout the course, directly applying them to a real-world cloud AI workflow:

### **✔ Cloud Computing**
- **Amazon S3** for cloud object storage of synthetic images and annotations  
- **AWS SageMaker** for model hosting, inference, and managed compute resources  
- **EC2** instances used by SageMaker endpoints under the hood  

### **✔ Virtualization & Scalability**
- SageMaker endpoints automatically scale based on incoming inference requests  
- S3 provides virtually unlimited storage for synthetic datasets  

### **✔ Reproducibility & CI/CD**
- **GitHub Actions** is used to automate:
  - Deployment of SageMaker inference endpoints  
  - Synchronization of data to S3  
  - Code version tracking and testing  
- Ensures full reproducibility of the deployment pipeline  

### **✔ Monitoring & Logging**
- Inference outputs and execution logs are captured through SageMaker  
- AWS CloudWatch supports monitoring endpoint behavior and performance  

### **✔ Ethical Considerations**
- Synthetic data avoids privacy issues associated with real-world datasets  
- Reduces reliance on human subjects and potential biases in manual annotations  

---

## 📌 3. Documentation & Workflow Explanation

### **Overall Workflow**

```txt
Isaac Sim (Replicator API)
        ↓
Synthetic Images + Annotations
        ↓ Export
Amazon S3 Bucket (Cloud Storage)
        ↓
AWS SageMaker (Pre-trained Model Endpoint)
        ↓
Inference on Synthetic Data
        ↓
Results Logged / Returned to User


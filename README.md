# AIMD-Sim-to-Cloud : Synthetic Data Pipeline for Computer Vision via AWS

This project explores the deployment of **synthetically generated data** for downstream computer vision tasks using **AWS cloud services**. Synthetic data is created using **NVIDIA Isaac Sim’s Replicator API**, after which the rendered images and annotations are exported to **Amazon S3** for scalable and secure cloud storage.

Once stored, the dataset is accessed by **AWS SageMaker**, which is used to train computer vision models using its built-in algorithms and managed compute services. To ensure automation, reproducibility, and continuous integration, a **GitHub Actions** workflow is incorporated to handle version control, testing, and model deployment steps.

---

## 🚀 Project Objectives

- Generate high-quality synthetic computer vision datasets using **Isaac Sim Replicator API**  
- Export images and annotations directly to **AWS S3**  
- Use **AWS SageMaker** to retrieve training data from S3 and run built-in ML algorithms  
- Integrate **GitHub Actions** for CI/CD, model deployment, and workflow automation  
- Apply concepts from cloud computing including:
  - **Version Control:** GitHub Actions  
  - **Virtualized Instances:** EC2  
  - **Cloud Storage:** S3  
  - **Model Training & Deployment:** SageMaker  

---

## 🧰 Technologies Used

- **NVIDIA Isaac Sim + Replicator API**
- **AWS S3**
- **AWS EC2**
- **AWS SageMaker**
- **GitHub Actions (CI/CD)**
- **Python**
- **Docker (optional)**

---

## 📦 High-Level Workflow

1. **Synthetic Data Generation**  
   Use Isaac Sim’s Replicator API to generate images and annotations.

2. **Cloud Storage with S3**  
   Export all generated data directly into an S3 bucket.

3. **Model Training in SageMaker**  
   Launch SageMaker training jobs that pull data from S3 and use built-in CV algorithms.

4. **Automation with GitHub Actions**  
   CI/CD pipeline handles:
   - Version control checks  
   - Automated builds/tests  
   - Deployment of new training jobs or models  

---

## 📁 Repository Structure (Suggested)

```txt
├── data/                  # Sample synthetic data (if small)
├── src/
│   ├── replicator/        # Isaac Sim scripts
│   ├── sagemaker/         # Training and deployment scripts
│   └── utils/             
├── .github/
│   └── workflows/         # GitHub Actions pipelines
├── docs/                  # Documentation
├── README.md              # Project overview
└── requirements.txt

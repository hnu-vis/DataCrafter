# Expansion Vis

This project consists of a frontend built with [Vite](https://vitejs.dev/), a backend built with [Flask](https://flask.palletsprojects.com/), and a demo environment. This README provides an overview of the project's structure, setup instructions, and how to run the demo.

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Running the Demo](#running-the-demo)
- [Project Structure](#project-structure)
- [Frontend Development](#frontend-development)
- [Backend Development](#backend-development)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
.
├── backend/         # Backend folder (Flask project)
├── frontend/        # Frontend folder (Vite project)
├── demo/            # Demo data folder
├── run_demo.sh      # Script to run both frontend and backend
└── README.md        # Project documentation
```

### Frontend

The `frontend` folder contains the Vite-based frontend code. It is responsible for the user interface and communicates with the Flask backend via HTTP requests.

### Backend

The `backend` folder contains the Flask-based backend code, which provides an API that the frontend can interact with. It also interacts with demo data for testing purposes.

### Demo

The `demo` folder holds sample data used to demonstrate the functionality of the frontend and backend without needing full deployment or database integration.

## Getting Started

### Requirements

- [Node.js](https://nodejs.org/) (for the frontend)
- [Python 3.x](https://www.python.org/) (for the backend)
- [Flask](https://flask.palletsprojects.com/) and other backend dependencies
- Bash terminal (for running `run_demo.sh`)

### Setup

1. **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <project-directory>
    ```

2. **Install Frontend Dependencies**
    ```bash
    cd frontend
    npm install
    ```

3. **Install Backend Dependencies**
    ```bash
    cd ../backend
    python3 -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### Running the Demo

To run the project with demo data, simply execute the `run_demo.sh` script from the project’s root directory:

```bash
./run_demo.sh
```

This script will:
- Start the Flask backend server.
- Start the Vite development server for the frontend.

After running the script, the project should be accessible at `http://localhost:3000` (or the port specified by the frontend configuration).

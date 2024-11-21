# Expansion Vis

## Project Structure

This project includes a front-end environment built with Vite and a back-end environment built with Flask. Before getting started, please download the [demo data](https://drive.google.com/file/d/1se-uJddNTuUKAenlMDAu4dL99Xi-YjrT/view?usp=drive_link) and place it in the root directory of this project.

```bash
.
├── backend/         # Backend folder (Flask project)
├── frontend/        # Frontend folder (Vite project)
├── demo/            # Demo data folder
├── run_demo.sh      # Script to run both frontend and backend
└── README.md        # Project documentation
```

## Frontend

1. Navigate to the `frontend` folder and install dependencies:

   ```bash
   npm install
   ```

2. Configure the static file directory, which is the location of the `demo` folder. In the `frontend/vite.config.ts` file, update the `publicDir` to the actual path of the `demo` folder.

3. Start the development server:

   ```bash
   npm run dev
   ```

   The frontend will run on `http://localhost:8081`.

## Backend

1. Navigate to the `backend` folder and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Fix a historical bug in the `flask_script` package: In the `__init__.py` file of this package, replace the line:

    ```python
    from flask._compat import text_type
    ```

    at line 15 with:

    ```python
    from flask_script._compat import text_type
    ```

3. Start the backend server:

   ```bash
   python manager.py run
   ```

   The backend will run on `http://localhost:8082`.

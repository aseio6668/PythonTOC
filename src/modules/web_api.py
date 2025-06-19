"""
Web API and Dashboard for Python to C++ Translator

Provides a REST API and web interface for cloud-based translation services.
Features:
- Project management
- Real-time translation
- Collaboration features
- Progress tracking
- File management
- Translation history
"""

import os
import sys
import json
import uuid
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
    from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session, relationship
    from sqlalchemy.sql import func
    WEB_DEPS_AVAILABLE = True
except ImportError:
    WEB_DEPS_AVAILABLE = False

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

if WEB_DEPS_AVAILABLE:
    from translator.translator import PythonToCppTranslator
    from modules.dependency_manager import DependencyManager
    from modules.dynamic_analyzer import DynamicModuleAnalyzer
    from modules.plugin_system import PluginManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine("sqlite:///translator_web.db", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Database Models
class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    owner_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    status = Column(String, default="active")
    
    files = relationship("ProjectFile", back_populates="project")
    translations = relationship("Translation", back_populates="project")


class ProjectFile(Base):
    __tablename__ = "project_files"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # 'python', 'cpp', 'header'
    content = Column(Text)
    size = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    project = relationship("Project", back_populates="files")


class Translation(Base):
    __tablename__ = "translations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    source_file_id = Column(String, ForeignKey("project_files.id"))
    target_file_id = Column(String, ForeignKey("project_files.id"), nullable=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    translation_options = Column(Text)  # JSON string
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    
    project = relationship("Project", back_populates="translations")


class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    api_key = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)


# Pydantic Models
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    owner_id: str
    created_at: datetime
    updated_at: datetime
    status: str
    file_count: int = 0
    translation_count: int = 0

    class Config:
        from_attributes = True


class FileUploadResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    size: int
    upload_time: datetime


class TranslationRequest(BaseModel):
    source_file_id: str
    options: Optional[Dict[str, Any]] = None
    use_plugins: bool = True
    optimization_level: str = "standard"


class TranslationResponse(BaseModel):
    id: str
    status: str
    progress: int
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class WebSocketManager:
    """Manager for WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)


class TranslationWorker:
    """Background worker for handling translations"""
    
    def __init__(self, db_session_factory, websocket_manager: WebSocketManager):
        self.db_session_factory = db_session_factory
        self.websocket_manager = websocket_manager
        self.plugin_manager = PluginManager()
        self.plugin_manager.load_plugins()
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """Start the background worker"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Translation worker started")
    
    def stop(self):
        """Stop the background worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Translation worker stopped")
    
    def _worker_loop(self):
        """Main worker loop"""
        while self.running:
            try:
                with self.db_session_factory() as db:
                    # Find pending translations
                    pending = db.query(Translation).filter(
                        Translation.status == "pending"
                    ).first()
                    
                    if pending:
                        self._process_translation(db, pending)
                
                # Sleep briefly to avoid busy waiting
                asyncio.run(asyncio.sleep(1))
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                asyncio.run(asyncio.sleep(5))
    
    def _process_translation(self, db: Session, translation: Translation):
        """Process a single translation"""
        try:
            # Update status
            translation.status = "processing"
            translation.progress = 10
            db.commit()
            
            # Notify via WebSocket
            asyncio.run(self.websocket_manager.send_personal_message({
                "type": "translation_progress",
                "translation_id": translation.id,
                "status": "processing",
                "progress": 10
            }, translation.project.owner_id))
            
            # Get source file
            source_file = db.query(ProjectFile).filter(
                ProjectFile.id == translation.source_file_id
            ).first()
            
            if not source_file:
                raise Exception("Source file not found")
            
            # Parse translation options
            options = {}
            if translation.translation_options:
                options = json.loads(translation.translation_options)
            
            translation.progress = 30
            db.commit()
            
            # Perform translation
            translator = PythonToCppTranslator()
            
            # Use plugin system if enabled
            if options.get("use_plugins", True):
                translator.plugin_manager = self.plugin_manager
            
            translation.progress = 50
            db.commit()
            
            # Translate the code
            cpp_code = translator.translate(source_file.content)
            
            translation.progress = 80
            db.commit()
            
            # Create target file
            target_filename = source_file.filename.replace('.py', '.cpp')
            target_file = ProjectFile(
                project_id=translation.project_id,
                filename=target_filename,
                file_path=f"generated/{target_filename}",
                file_type="cpp",
                content=cpp_code,
                size=len(cpp_code.encode('utf-8'))
            )
            
            db.add(target_file)
            db.flush()
            
            # Update translation
            translation.target_file_id = target_file.id
            translation.status = "completed"
            translation.progress = 100
            translation.completed_at = func.now()
            
            db.commit()
            
            # Notify completion
            asyncio.run(self.websocket_manager.send_personal_message({
                "type": "translation_complete",
                "translation_id": translation.id,
                "target_file_id": target_file.id,
                "status": "completed"
            }, translation.project.owner_id))
            
            logger.info(f"Translation completed: {translation.id}")
            
        except Exception as e:
            # Mark as failed
            translation.status = "failed"
            translation.error_message = str(e)
            db.commit()
            
            # Notify failure
            asyncio.run(self.websocket_manager.send_personal_message({
                "type": "translation_error",
                "translation_id": translation.id,
                "error": str(e),
                "status": "failed"
            }, translation.project.owner_id))
            
            logger.error(f"Translation failed: {translation.id} - {e}")


# Initialize FastAPI app
if WEB_DEPS_AVAILABLE:
    app = FastAPI(
        title="Python to C++ Translator API",
        description="REST API for Python to C++ translation services",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize managers
    websocket_manager = WebSocketManager()
    translation_worker = TranslationWorker(SessionLocal, websocket_manager)
    
    # Security
    security = HTTPBearer()
    
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
        """Get current user from API key"""
        user = db.query(User).filter(User.api_key == credentials.credentials).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Update last active
        user.last_active = func.now()
        db.commit()
        
        return user
    
    # Static files (for web dashboard)
    app.mount("/static", StaticFiles(directory="web/static"), name="static")
    
    # API Routes
    @app.get("/")
    async def root():
        return {"message": "Python to C++ Translator API", "version": "1.0.0"}
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve the web dashboard"""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Python to C++ Translator</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
        </head>
        <body class="bg-gray-100">
            <div id="app" class="container mx-auto p-4">
                <h1 class="text-3xl font-bold mb-6">Python to C++ Translator</h1>
                
                <!-- Project List -->
                <div class="bg-white rounded-lg shadow p-6 mb-6">
                    <h2 class="text-xl font-semibold mb-4">Projects</h2>
                    <button @click="showCreateProject = true" class="bg-blue-500 text-white px-4 py-2 rounded mb-4">
                        New Project
                    </button>
                    
                    <div v-if="projects.length === 0" class="text-gray-500">
                        No projects yet. Create your first project!
                    </div>
                    
                    <div v-for="project in projects" :key="project.id" class="border rounded p-4 mb-2">
                        <h3 class="font-semibold">{{ project.name }}</h3>
                        <p class="text-gray-600">{{ project.description }}</p>
                        <div class="mt-2 text-sm text-gray-500">
                            Files: {{ project.file_count }} | Translations: {{ project.translation_count }}
                        </div>
                        <button @click="selectProject(project)" class="mt-2 bg-green-500 text-white px-3 py-1 rounded text-sm">
                            Open
                        </button>
                    </div>
                </div>
                
                <!-- Selected Project -->
                <div v-if="selectedProject" class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-semibold mb-4">{{ selectedProject.name }}</h2>
                    
                    <!-- File Upload -->
                    <div class="mb-4">
                        <input type="file" @change="handleFileUpload" accept=".py" class="mb-2">
                        <button @click="uploadFile" :disabled="!selectedFile" class="bg-blue-500 text-white px-4 py-2 rounded">
                            Upload Python File
                        </button>
                    </div>
                    
                    <!-- Files List -->
                    <div class="mb-6">
                        <h3 class="font-semibold mb-2">Files</h3>
                        <div v-for="file in projectFiles" :key="file.id" class="flex justify-between items-center border rounded p-2 mb-1">
                            <span>{{ file.filename }} ({{ file.file_type }})</span>
                            <button v-if="file.file_type === 'python'" @click="translateFile(file)" class="bg-purple-500 text-white px-3 py-1 rounded text-sm">
                                Translate
                            </button>
                        </div>
                    </div>
                    
                    <!-- Translations -->
                    <div>
                        <h3 class="font-semibold mb-2">Translations</h3>
                        <div v-for="translation in translations" :key="translation.id" class="border rounded p-2 mb-1">
                            <div class="flex justify-between items-center">
                                <span>Translation {{ translation.id.substr(0, 8) }}</span>
                                <span class="px-2 py-1 rounded text-sm" :class="getStatusClass(translation.status)">
                                    {{ translation.status }}
                                </span>
                            </div>
                            <div v-if="translation.status === 'processing'" class="mt-2">
                                <div class="bg-gray-200 rounded-full h-2">
                                    <div class="bg-blue-500 h-2 rounded-full" :style="{width: translation.progress + '%'}"></div>
                                </div>
                                <div class="text-sm text-gray-600 mt-1">{{ translation.progress }}%</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Create Project Modal -->
                <div v-if="showCreateProject" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                    <div class="bg-white rounded-lg p-6 w-96">
                        <h3 class="text-lg font-semibold mb-4">Create New Project</h3>
                        <input v-model="newProject.name" placeholder="Project Name" class="w-full border rounded p-2 mb-2">
                        <textarea v-model="newProject.description" placeholder="Description" class="w-full border rounded p-2 mb-4"></textarea>
                        <div class="flex justify-end space-x-2">
                            <button @click="showCreateProject = false" class="bg-gray-500 text-white px-4 py-2 rounded">
                                Cancel
                            </button>
                            <button @click="createProject" class="bg-blue-500 text-white px-4 py-2 rounded">
                                Create
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <script>
            const { createApp } = Vue

            createApp({
                data() {
                    return {
                        projects: [],
                        selectedProject: null,
                        projectFiles: [],
                        translations: [],
                        showCreateProject: false,
                        newProject: { name: '', description: '' },
                        selectedFile: null,
                        ws: null,
                        apiKey: 'demo-api-key'
                    }
                },
                mounted() {
                    this.loadProjects()
                    this.connectWebSocket()
                },
                methods: {
                    async apiCall(url, options = {}) {
                        const response = await fetch(url, {
                            ...options,
                            headers: {
                                'Authorization': `Bearer ${this.apiKey}`,
                                'Content-Type': 'application/json',
                                ...options.headers
                            }
                        })
                        
                        if (!response.ok) {
                            throw new Error('API call failed')
                        }
                        
                        return response.json()
                    },
                    
                    async loadProjects() {
                        try {
                            this.projects = await this.apiCall('/api/projects')
                        } catch (error) {
                            console.error('Failed to load projects:', error)
                        }
                    },
                    
                    async createProject() {
                        try {
                            await this.apiCall('/api/projects', {
                                method: 'POST',
                                body: JSON.stringify(this.newProject)
                            })
                            
                            this.showCreateProject = false
                            this.newProject = { name: '', description: '' }
                            this.loadProjects()
                        } catch (error) {
                            console.error('Failed to create project:', error)
                        }
                    },
                    
                    async selectProject(project) {
                        this.selectedProject = project
                        await this.loadProjectFiles()
                        await this.loadTranslations()
                    },
                    
                    async loadProjectFiles() {
                        try {
                            this.projectFiles = await this.apiCall(`/api/projects/${this.selectedProject.id}/files`)
                        } catch (error) {
                            console.error('Failed to load files:', error)
                        }
                    },
                    
                    async loadTranslations() {
                        try {
                            this.translations = await this.apiCall(`/api/projects/${this.selectedProject.id}/translations`)
                        } catch (error) {
                            console.error('Failed to load translations:', error)
                        }
                    },
                    
                    handleFileUpload(event) {
                        this.selectedFile = event.target.files[0]
                    },
                    
                    async uploadFile() {
                        if (!this.selectedFile) return
                        
                        const formData = new FormData()
                        formData.append('file', this.selectedFile)
                        
                        try {
                            await fetch(`/api/projects/${this.selectedProject.id}/files`, {
                                method: 'POST',
                                headers: {
                                    'Authorization': `Bearer ${this.apiKey}`
                                },
                                body: formData
                            })
                            
                            this.selectedFile = null
                            this.loadProjectFiles()
                        } catch (error) {
                            console.error('Failed to upload file:', error)
                        }
                    },
                    
                    async translateFile(file) {
                        try {
                            await this.apiCall(`/api/translate`, {
                                method: 'POST',
                                body: JSON.stringify({
                                    source_file_id: file.id,
                                    options: { use_plugins: true }
                                })
                            })
                            
                            this.loadTranslations()
                        } catch (error) {
                            console.error('Failed to start translation:', error)
                        }
                    },
                    
                    getStatusClass(status) {
                        const classes = {
                            pending: 'bg-yellow-200 text-yellow-800',
                            processing: 'bg-blue-200 text-blue-800',
                            completed: 'bg-green-200 text-green-800',
                            failed: 'bg-red-200 text-red-800'
                        }
                        return classes[status] || 'bg-gray-200 text-gray-800'
                    },
                    
                    connectWebSocket() {
                        this.ws = new WebSocket(`ws://localhost:8000/ws/updates`)
                        
                        this.ws.onmessage = (event) => {
                            const data = JSON.parse(event.data)
                            
                            if (data.type === 'translation_progress' || data.type === 'translation_complete') {
                                this.loadTranslations()
                            }
                        }
                        
                        this.ws.onclose = () => {
                            setTimeout(this.connectWebSocket, 5000)
                        }
                    }
                }
            }).mount('#app')
            </script>
        </body>
        </html>
        """
        return HTMLResponse(dashboard_html)
    
    @app.post("/api/projects", response_model=ProjectResponse)
    async def create_project(
        project: ProjectCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Create a new project"""
        db_project = Project(
            name=project.name,
            description=project.description,
            owner_id=current_user.id
        )
        
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        
        return ProjectResponse(
            id=db_project.id,
            name=db_project.name,
            description=db_project.description,
            owner_id=db_project.owner_id,
            created_at=db_project.created_at,
            updated_at=db_project.updated_at,
            status=db_project.status,
            file_count=0,
            translation_count=0
        )
    
    @app.get("/api/projects", response_model=List[ProjectResponse])
    async def list_projects(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """List user's projects"""
        projects = db.query(Project).filter(
            Project.owner_id == current_user.id,
            Project.status == "active"
        ).all()
        
        result = []
        for project in projects:
            file_count = db.query(ProjectFile).filter(ProjectFile.project_id == project.id).count()
            translation_count = db.query(Translation).filter(Translation.project_id == project.id).count()
            
            result.append(ProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                owner_id=project.owner_id,
                created_at=project.created_at,
                updated_at=project.updated_at,
                status=project.status,
                file_count=file_count,
                translation_count=translation_count
            ))
        
        return result
    
    @app.post("/api/projects/{project_id}/files", response_model=FileUploadResponse)
    async def upload_file(
        project_id: str,
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Upload a file to a project"""
        # Verify project ownership
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Determine file type
        file_type = "python" if file.filename.endswith('.py') else "other"
        
        # Create file record
        db_file = ProjectFile(
            project_id=project_id,
            filename=file.filename,
            file_path=f"uploads/{file.filename}",
            file_type=file_type,
            content=content_str,
            size=len(content)
        )
        
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        return FileUploadResponse(
            id=db_file.id,
            filename=db_file.filename,
            file_type=db_file.file_type,
            size=db_file.size,
            upload_time=db_file.created_at
        )
    
    @app.get("/api/projects/{project_id}/files")
    async def list_project_files(
        project_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """List files in a project"""
        # Verify project ownership
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        files = db.query(ProjectFile).filter(ProjectFile.project_id == project_id).all()
        
        return [
            {
                "id": f.id,
                "filename": f.filename,
                "file_type": f.file_type,
                "size": f.size,
                "created_at": f.created_at
            }
            for f in files
        ]
    
    @app.post("/api/translate", response_model=TranslationResponse)
    async def start_translation(
        request: TranslationRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Start a translation job"""
        # Verify file ownership
        source_file = db.query(ProjectFile).filter(ProjectFile.id == request.source_file_id).first()
        if not source_file:
            raise HTTPException(status_code=404, detail="Source file not found")
        
        project = db.query(Project).filter(
            Project.id == source_file.project_id,
            Project.owner_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create translation job
        translation = Translation(
            project_id=source_file.project_id,
            source_file_id=request.source_file_id,
            translation_options=json.dumps(request.options or {})
        )
        
        db.add(translation)
        db.commit()
        db.refresh(translation)
        
        return TranslationResponse(
            id=translation.id,
            status=translation.status,
            progress=translation.progress,
            started_at=translation.started_at,
            completed_at=translation.completed_at,
            error_message=translation.error_message
        )
    
    @app.get("/api/projects/{project_id}/translations")
    async def list_translations(
        project_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """List translations for a project"""
        # Verify project ownership
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.owner_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        translations = db.query(Translation).filter(Translation.project_id == project_id).all()
        
        return [
            {
                "id": t.id,
                "status": t.status,
                "progress": t.progress,
                "started_at": t.started_at,
                "completed_at": t.completed_at,
                "error_message": t.error_message
            }
            for t in translations
        ]
    
    @app.get("/api/files/{file_id}/download")
    async def download_file(
        file_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Download a file"""
        file_record = db.query(ProjectFile).filter(ProjectFile.id == file_id).first()
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        project = db.query(Project).filter(
            Project.id == file_record.project_id,
            Project.owner_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix=f"_{file_record.filename}", delete=False) as tmp:
            tmp.write(file_record.content)
            tmp_path = tmp.name
        
        return FileResponse(
            tmp_path,
            filename=file_record.filename,
            media_type='application/octet-stream'
        )
    
    @app.websocket("/ws/updates")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        client_id = str(uuid.uuid4())
        await websocket_manager.connect(websocket, client_id)
        
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            websocket_manager.disconnect(client_id)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize database and start background worker"""
        Base.metadata.create_all(bind=engine)
        
        # Create demo user if not exists
        with SessionLocal() as db:
            demo_user = db.query(User).filter(User.username == "demo").first()
            if not demo_user:
                demo_user = User(
                    username="demo",
                    email="demo@example.com",
                    api_key="demo-api-key"
                )
                db.add(demo_user)
                db.commit()
        
        translation_worker.start()
        logger.info("Web API started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        translation_worker.stop()
        logger.info("Web API stopped")


def create_web_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Create and configure the web server"""
    if not WEB_DEPS_AVAILABLE:
        print("Web dependencies not available. Install with:")
        print("pip install fastapi uvicorn sqlalchemy pydantic")
        return None
    
    return uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def main():
    """Run the web server"""
    if not WEB_DEPS_AVAILABLE:
        print("Missing dependencies for web API. Install with:")
        print("pip install fastapi uvicorn sqlalchemy pydantic websockets")
        sys.exit(1)
    
    print("Starting Python to C++ Translator Web API...")
    print(f"Dashboard will be available at: http://localhost:8000/dashboard")
    print(f"API documentation at: http://localhost:8000/docs")
    
    config = create_web_server(reload=True)
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()

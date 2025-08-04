import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variables for zoom, drag and camera
current_image = None
current_image_index = -1
image_list = []  # List to store all loaded images
zoom_factor = 1.0
drag_data = {"x": 0, "y": 0, "item": None}
camera_active = False
camera = None

# Function to detect objects in an image
def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return image

# Open multiple image files
def open_images():
    global current_image, current_image_index, image_list
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        image_list = []
        for file_path in file_paths:
            image = cv2.imread(file_path)
            if image is not None:
                image = detect_objects(image)
                image_list.append(image)
        
        if image_list:
            current_image_index = 0
            current_image = image_list[0]
            show_image(current_image)
            update_navigation_buttons()

def next_image():
    global current_image_index, current_image
    if image_list and current_image_index < len(image_list) - 1:
        current_image_index += 1
        current_image = image_list[current_image_index]
        show_image(current_image)
        update_navigation_buttons()

def previous_image():
    global current_image_index, current_image
    if image_list and current_image_index > 0:
        current_image_index -= 1
        current_image = image_list[current_image_index]
        show_image(current_image)
        update_navigation_buttons()

def update_navigation_buttons():
    # Update navigation buttons state
    btn_prev.config(state=tk.NORMAL if current_image_index > 0 else tk.DISABLED)
    btn_next.config(state=tk.NORMAL if current_image_index < len(image_list) - 1 else tk.DISABLED)
    # Update image counter label
    if image_list:
        counter_label.config(text=f"Image {current_image_index + 1} of {len(image_list)}")
    else:
        counter_label.config(text="No images loaded")

# Mouse event handlers for dragging
def start_drag(event):
    # Record the starting position and item
    drag_data["x"] = event.x
    drag_data["y"] = event.y
    drag_data["item"] = canvas.find_closest(event.x, event.y)[0]
    canvas.config(cursor="hand2")  # Change cursor to hand during drag

def stop_drag(event):
    # Reset the drag data
    drag_data["item"] = None
    drag_data["x"] = 0
    drag_data["y"] = 0
    canvas.config(cursor="arrow")  # Reset cursor

def drag(event):
    # Calculate how far the mouse has moved
    if drag_data["item"]:
        dx = event.x - drag_data["x"]
        dy = event.y - drag_data["y"]
        
        # Move the image
        canvas.move(drag_data["item"], dx, dy)
        
        # Update the stored position
        drag_data["x"] = event.x
        drag_data["y"] = event.y

# Open camera feed
def start_camera():
    global current_image, camera_active, camera
    if not camera_active:
        camera = cv2.VideoCapture(0)
        camera_active = True
        
        def update_camera():
            global camera_active
            if camera_active and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    frame = detect_objects(frame)
                    show_camera_frame(frame)
                    root.after(10, update_camera)  # Update every 10ms
                else:
                    stop_camera()
            
        update_camera()

# Show camera frame (without zoom)
def show_camera_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Resize image to fit canvas while maintaining aspect ratio
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    if canvas_width <= 1 or canvas_height <= 1:
        canvas_width = 800
        canvas_height = 600
    
    # Calculate scaling to fit the window while maintaining aspect ratio
    width, height = image.size
    scale = min(canvas_width/width, canvas_height/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    
    canvas.delete("all")
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo

def stop_camera():
    global camera_active, camera, current_image
    if camera_active:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None
        current_image = None
        canvas.delete("all")

# Show image with zoom
def show_image(frame, apply_zoom=True):
    global current_image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    if apply_zoom:
        # Get original size
        width, height = image.size
        # Calculate new size based on zoom factor
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    photo = ImageTk.PhotoImage(image)
    canvas.delete("all")
    # Create image at the current drag position
    canvas.create_image(drag_data["x"], drag_data["y"], image=photo, anchor="nw")
    canvas.image = photo

def zoom_in():
    global zoom_factor, current_image
    if current_image is not None:
        zoom_factor *= 1.2
        show_image(current_image)

def zoom_out():
    global zoom_factor, current_image
    if current_image is not None:
        zoom_factor /= 1.2
        show_image(current_image)

def clear_image():
    global current_image, zoom_factor, drag_data, image_list, current_image_index
    # Reset all global variables
    current_image = None
    current_image_index = -1
    image_list = []
    zoom_factor = 1.0
    drag_data = {"x": 0, "y": 0, "item": None}
    stop_camera()  # Make sure to stop camera if it's running
    # Clear the canvas
    canvas.delete("all")
    update_navigation_buttons()

# Modify the quit function to properly close everything
def quit_app():
    stop_camera()  # Make sure to stop camera before quitting
    root.quit()

# Setup GUI
root = tk.Tk()
root.title("YOLO Object Detection")
root.geometry("1024x768")  # Set initial window size

# Create main container frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create button frame
button_frame = tk.Frame(main_frame)
button_frame.pack(fill=tk.X, padx=5, pady=5)

# Create navigation frame
nav_frame = tk.Frame(main_frame)
nav_frame.pack(fill=tk.X, padx=5, pady=5)

# Create canvas frame
canvas_frame = tk.Frame(main_frame)
canvas_frame.pack(fill=tk.BOTH, expand=True)

# Create canvas for displaying images
canvas = tk.Canvas(canvas_frame, bg='gray')
canvas.pack(fill=tk.BOTH, expand=True)

# Bind mouse events for dragging
canvas.bind("<Button-1>", start_drag)        # Left mouse button press
canvas.bind("<B1-Motion>", drag)             # Mouse motion while left button pressed
canvas.bind("<ButtonRelease-1>", stop_drag)  # Left mouse button release

# Add buttons
btn_image = tk.Button(button_frame, text="Load Images", command=open_images)
btn_image.pack(side=tk.LEFT, padx=5)

btn_camera = tk.Button(button_frame, text="Start Camera", command=start_camera)
btn_camera.pack(side=tk.LEFT, padx=5)

btn_stop_camera = tk.Button(button_frame, text="Stop Camera", command=stop_camera)
btn_stop_camera.pack(side=tk.LEFT, padx=5)

btn_zoom_in = tk.Button(button_frame, text="Zoom In", command=zoom_in)
btn_zoom_in.pack(side=tk.LEFT, padx=5)

btn_zoom_out = tk.Button(button_frame, text="Zoom Out", command=zoom_out)
btn_zoom_out.pack(side=tk.LEFT, padx=5)

btn_clear = tk.Button(button_frame, text="Clear", command=clear_image)
btn_clear.pack(side=tk.LEFT, padx=5)

btn_quit = tk.Button(button_frame, text="Quit", command=quit_app)
btn_quit.pack(side=tk.LEFT, padx=5)

# Add navigation buttons
btn_prev = tk.Button(nav_frame, text="Previous", command=previous_image, state=tk.DISABLED)
btn_prev.pack(side=tk.LEFT, padx=5)

counter_label = tk.Label(nav_frame, text="No images loaded")
counter_label.pack(side=tk.LEFT, padx=5)

btn_next = tk.Button(nav_frame, text="Next", command=next_image, state=tk.DISABLED)
btn_next.pack(side=tk.LEFT, padx=5)

# Ensure camera is released when window is closed
root.protocol("WM_DELETE_WINDOW", quit_app)

# Start GUI loop
root.mainloop() 
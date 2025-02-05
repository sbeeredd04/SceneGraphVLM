# Import required libraries
import os
import json
import pickle
from glob import glob
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import base64
import io
import traceback

def setup_environment():
    """Setup environment and BLIP model configurations"""
    try:
        print("Starting BLIP model setup...")
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load model and processor
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16
        )
        model.to(device)
        
        print("BLIP model setup completed")
        return model, processor, device
    except Exception as e:
        print(f"Model setup failed: {str(e)}")
        raise

def load_annotations():
    """Load annotation files and object/relationship classes"""
    print("Loading annotation files...")
    try:
        # Load object bbox and relationship data
        with open('/content/drive/MyDrive/data/object_bbox_and_relationship.pkl', 'rb') as f:
            bbox_rel_data = pickle.load(f)
        
        # Load object classes
        with open('/content/drive/MyDrive/data/object_classes.txt', 'r') as f:
            object_classes = f.read().splitlines()
            
        # Load relationship classes
        with open('/content/drive/MyDrive/data/relationship_classes.txt', 'r') as f:
            relationship_classes = f.read().splitlines()
            
        print(f"Loaded annotations with {len(object_classes)} objects, {len(relationship_classes)} relationships")
        return bbox_rel_data, object_classes, relationship_classes
    
    except Exception as e:
        print(f"Failed to load annotations: {str(e)}")
        raise

def process_video_frames(video_id, bbox_rel_data):
    """Process frames from a specific video folder"""
    try:
        print(f"Processing video: {video_id}")
        
        # Get all frames for this video
        frame_path = f"/content/drive/MyDrive/data/frames/{video_id}/*.png"
        frame_files = sorted(glob(frame_path))
        
        if not frame_files:
            print(f"No frames found in {frame_path}")
            return None
            
        print(f"Found {len(frame_files)} frames for video {video_id}")
        
        # Extract scene graphs
        scene_graphs = []
        for frame_file in tqdm(frame_files, desc="Extracting scene graphs"):
            # Construct the frame ID as it appears in the pkl file
            frame_name = os.path.basename(frame_file)
            frame_id = f"{video_id}/{frame_name}"
            
            # Get scene graph from annotations
            scene_graph = extract_scene_graph(frame_id, bbox_rel_data)
            
            if scene_graph:
                print(f"Found scene graph for frame {frame_id}")
                scene_graphs.append({
                    'frame_id': frame_id,
                    'scene_graph': scene_graph
                })
            else:
                print(f"No scene graph found for frame {frame_id}")
        
        print(f"Extracted {len(scene_graphs)} scene graphs")
        return scene_graphs
    
    except Exception as e:
        print(f"Video processing failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def extract_scene_graph(frame_id, bbox_rel_data):
    """Extract scene graph from annotations"""
    try:
        # Get annotations for this frame
        frame_anns = bbox_rel_data.get(frame_id, [])
        if not frame_anns:
            print(f"No annotations found for frame {frame_id}")
            return None
            
        scene_graph = {
            "objects": [],
            "relationships": []
        }
        
        added_objects = set()
        
        for ann in frame_anns:
            if not ann.get('visible', True):
                continue
                
            obj_class = ann['class']
            if obj_class not in added_objects:
                scene_graph["objects"].append({
                    "object": obj_class,
                    "attributes": [],
                    "bbox": ann.get('bbox', [])
                })
                added_objects.add(obj_class)
            
            # Add relationships
            for rel_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
                for rel in ann.get(rel_type, []):
                    scene_graph["relationships"].append({
                        "subject": "person",
                        "predicate": rel,
                        "object": obj_class
                    })
        
        if scene_graph["objects"]:
            print(f"Extracted scene graph with {len(scene_graph['objects'])} objects and {len(scene_graph['relationships'])} relationships")
            return scene_graph
        else:
            print(f"No valid objects found in annotations for frame {frame_id}")
            return None
    
    except Exception as e:
        print(f"Failed to extract scene graph for frame {frame_id}: {str(e)}")
        return None

def print_scene_graph(scene_graph):
    """Pretty print scene graph"""
    if not scene_graph:
        print("No scene graph available")
        return
        
    print("\nObjects:")
    for obj in scene_graph["objects"]:
        attrs = ", ".join(obj.get("attributes", []))
        bbox = obj.get("bbox", [])
        print(f"- {obj['object']}" + 
              (f" ({attrs})" if attrs else "") +
              (f" bbox: {bbox}" if bbox else ""))
    
    print("\nRelationships:")
    for rel in scene_graph["relationships"]:
        print(f"- {rel['subject']} {rel['predicate']} {rel['object']}")

def split_scene_graphs(scene_graphs, split_ratio=0.8):
    """Split scene graphs into training and testing sets chronologically"""
    try:
        # Sort by frame number to ensure chronological order
        scene_graphs.sort(key=lambda x: int(x['frame_id'].split('/')[-1].split('.')[0]))
        
        split_idx = int(len(scene_graphs) * split_ratio)
        train_graphs = scene_graphs[:split_idx]
        test_graphs = scene_graphs[split_idx:]
        
        print(f"Split {len(scene_graphs)} graphs chronologically into {len(train_graphs)} train and {len(test_graphs)} test")
        print(f"Training frames: {train_graphs[0]['frame_id']} to {train_graphs[-1]['frame_id']}")
        print(f"Testing frames: {test_graphs[0]['frame_id']} to {test_graphs[-1]['frame_id']}")
        
        return train_graphs, test_graphs
    except Exception as e:
        print(f"Failed to split scene graphs: {str(e)}")
        raise

def predict_scene_graphs(train_graphs, num_predictions, model, processor, device,
                        vocabulary_type='closed', object_classes=None, relationship_classes=None):
    """Generate scene graph predictions using hybrid BLIP + Gemini approach with closed vocabulary"""
    try:
        print(f"\n=== Starting Hybrid Scene Graph Prediction (Closed Vocabulary) ===")
        print(f"Will generate {num_predictions} sequential predictions")
        print(f"Using {len(object_classes)} objects and {len(relationship_classes)} relationships")
        
        # Print available vocabulary
        print("\nAvailable Objects:", ", ".join(object_classes))
        print("Available Relationships:", ", ".join(relationship_classes))
        
        predictions = []
        context_graphs = train_graphs.copy()
        
        # Get test frame IDs to predict
        last_train_num = int(context_graphs[-1]['frame_id'].split('/')[-1].split('.')[0])
        video_id = context_graphs[0]['frame_id'].split('/')[0]
        
        # Find next frames to predict
        frame_pattern = f"/content/drive/MyDrive/data/frames/{video_id}/*.png"
        all_frames = sorted(glob(frame_pattern))
        test_frame_nums = []
        
        for frame_path in all_frames:
            frame_num = int(frame_path.split('/')[-1].split('.')[0])
            if frame_num > last_train_num:
                test_frame_nums.append(frame_num)
        
        # Generate predictions for each test frame
        for i, next_frame_num in enumerate(test_frame_nums[:num_predictions], 1):
            print(f"\n=== PREDICTION {i}/{num_predictions} ===")
            print(f"Predicting frame {video_id}/{next_frame_num:06d}.png")
            
            # Load previous frame image for context
            prev_frame_num = int(context_graphs[-1]['frame_id'].split('/')[-1].split('.')[0])
            prev_frame_path = f"/content/drive/MyDrive/data/frames/{video_id}/{prev_frame_num:06d}.png"
            print(f"Using previous frame for context: {prev_frame_path}")
            image = Image.open(prev_frame_path).convert('RGB')
            
            # Prepare context from previous frames
            context_str = "Based on the previous frames:\n"
            for ctx in context_graphs[-3:]:  # Use last 3 frames for context
                objects = [obj['object'] for obj in ctx['scene_graph']['objects']]
                relationships = [f"{r['subject']} {r['predicate']} {r['object']}" 
                               for r in ctx['scene_graph']['relationships']]
                context_str += f"\nFrame {ctx['frame_id']}:\n"
                context_str += f"Objects: {', '.join(objects)}\n"
                context_str += f"Relationships: {', '.join(relationships)}\n"
            
            # Process with BLIP
            inputs = processor(image, text=context_str, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            blip_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            print("\n=== BLIP RESPONSE ===")
            print(blip_response)
            
            # Update BLIP prompt to include vocabulary constraints
            prompt = f"""
            {context_str}
            
            Based on this image and the context above, predict what objects and relationships will appear in the next frame.
            
            Use ONLY these objects: {', '.join(object_classes)}
            Use ONLY these relationships: {', '.join(relationship_classes)}
            
            Focus on:
            1. Objects that are clearly present
            2. Direct relationships between objects
            3. Only use the provided vocabulary terms
            
            Be specific and natural in your description.
            """
            
            print("\n=== BLIP PROMPT ===")
            print(prompt)
            
            # Update Gemini prompt to enforce vocabulary
            gemini_prompt = f"""
            Analyze this scene description and previous frames:

            {blip_response}

            Generate a scene graph prediction using ONLY these terms:
            
            Objects: {', '.join(object_classes)}
            Relationships: {', '.join(relationship_classes)}

            The scene graph must follow this exact structure:
            {{
                "objects": [
                    {{"object": "object_name"}}
                ],
                "relationships": [
                    {{"subject": "subject_name", "predicate": "predicate_name", "object": "object_name"}}
                ]
            }}

            Rules:
            1. Only use objects from the provided list
            2. Only use relationships from the provided list
            3. Every relationship must have a valid subject and object
            4. Output only valid JSON, no other text
            """
            
            print("\n=== GEMINI PROMPT ===")
            print(gemini_prompt)
            
            # Get Gemini response and validate against vocabulary
            gemini_response = get_gemini_response(gemini_prompt)
            
            print("\n=== GEMINI RESPONSE ===")
            print(gemini_response)
            
            try:
                predicted_graph = json.loads(gemini_response)
                # Validate against closed vocabulary
                predicted_graph = validate_vocabulary(
                    predicted_graph, 
                    object_classes, 
                    relationship_classes,
                    strict=True  # Enforce strict vocabulary matching
                )
                
                print("\n=== VALIDATED SCENE GRAPH ===")
                print(f"Objects ({len(predicted_graph['objects'])}):")
                for obj in predicted_graph['objects']:
                    print(f"- {obj['object']}")
                print(f"\nRelationships ({len(predicted_graph['relationships'])}):")
                for rel in predicted_graph['relationships']:
                    print(f"- {rel['subject']} {rel['predicate']} {rel['object']}")
                
            except Exception as e:
                print(f"\nFailed to parse or validate scene graph: {str(e)}")
                predicted_graph = {
                    "objects": [{"object": "person"}],
                    "relationships": []
                }
            
            # Add prediction
            prediction_entry = {
                'frame_id': f"{video_id}/{next_frame_num:06d}.png",
                'scene_graph': predicted_graph
            }
            predictions.append(prediction_entry)
            context_graphs.append(prediction_entry)
            
        return predictions
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def visualize_scene_graph(scene_graph, title, ax):
    """Visualize scene graph using networkx"""
    try:
        if not scene_graph or 'objects' not in scene_graph:
            print(f"Invalid scene graph structure for {title}: {scene_graph}")
            return
            
        G = nx.Graph()
        
        # Add object nodes
        for obj in scene_graph["objects"]:
            if isinstance(obj, dict) and "object" in obj:
                G.add_node(obj["object"], type="object")
            else:
                print(f"Invalid object structure: {obj}")
                continue
            
        # Add relationship edges
        for rel in scene_graph.get("relationships", []):
            if isinstance(rel, dict) and all(k in rel for k in ["subject", "predicate", "object"]):
                G.add_edge(rel["subject"], rel["object"], 
                          label=rel["predicate"])
            else:
                print(f"Invalid relationship structure: {rel}")
                continue
            
        # Draw graph
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=1000, ax=ax)
            nx.draw_networkx_labels(G, pos, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax)
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        else:
            print(f"No valid nodes to visualize for {title}")
            
        ax.set_title(title)
        ax.axis('off')
        
    except Exception as e:
        print(f"Failed to visualize scene graph: {str(e)}")
        print(f"Scene graph: {scene_graph}")
        raise

def visualize_comparison(frame_path, ground_truth, prediction):
    """Visualize frame image with ground truth and predicted graphs"""
    try:
        fig = plt.figure(figsize=(15, 5))
        
        # Show frame image
        ax1 = fig.add_subplot(131)
        img = Image.open(frame_path)
        ax1.imshow(img)
        ax1.set_title("Frame")
        ax1.axis('off')
        
        # Show ground truth graph
        ax2 = fig.add_subplot(132)
        visualize_scene_graph(ground_truth, "Ground Truth", ax2)
        
        # Show predicted graph
        ax3 = fig.add_subplot(133)
        visualize_scene_graph(prediction, "Prediction", ax3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        raise

def visualize_results(test_graphs, predictions):
    """Visualize all results"""
    try:
        if not predictions:
            print("No predictions to visualize")
            return
            
        print("Visualizing results...")
        for i, (test_graph, pred_graph) in enumerate(zip(test_graphs, predictions)):
            try:
                frame_path = f"/content/drive/MyDrive/data/frames/{test_graph['frame_id']}"
                print(f"Visualizing comparison for frame {test_graph['frame_id']}")
                
                # Extract scene graph from prediction (it's nested in the prediction structure)
                pred_scene_graph = pred_graph['scene_graph'] if isinstance(pred_graph, dict) else pred_graph
                
                # Log the structures for debugging
                print(f"Ground truth structure: {test_graph['scene_graph'].keys()}")
                print(f"Prediction structure: {pred_scene_graph.keys()}")
                
                visualize_comparison(
                    frame_path,
                    test_graph['scene_graph'],
                    pred_scene_graph
                )
            except Exception as e:
                print(f"Failed to visualize comparison {i+1}: {str(e)}")
                print(f"Ground truth: {test_graph}")
                print(f"Prediction: {pred_graph}")
                continue
                
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        raise

def validate_vocabulary(scene_graph, object_classes, relationship_classes, strict=True):
    """Validate and normalize scene graph vocabulary against allowed classes"""
    try:
        validated_graph = {
            "objects": [],
            "relationships": []
        }
        
        def validate_term(word, vocab):
            if strict:
                # In strict mode, only exact matches are allowed
                if word not in vocab:
                    raise ValueError(f"Term '{word}' not in vocabulary: {vocab}")
                return word
            else:
                # In non-strict mode, try to find closest match
                if word.lower() in [v.lower() for v in vocab]:
                    return vocab[[v.lower() for v in vocab].index(word.lower())]
                raise ValueError(f"No match found for '{word}' in vocabulary")
        
        # Validate objects
        seen_objects = set()
        for obj in scene_graph.get("objects", []):
            if isinstance(obj, dict) and "object" in obj:
                obj_name = validate_term(obj["object"], object_classes)
                if obj_name not in seen_objects:
                    validated_graph["objects"].append({
                        "object": obj_name,
                        "attributes": obj.get("attributes", []),
                        "bbox": obj.get("bbox", [])
                    })
                    seen_objects.add(obj_name)
        
        # Validate relationships
        for rel in scene_graph.get("relationships", []):
            if isinstance(rel, dict) and all(k in rel for k in ["subject", "predicate", "object"]):
                validated_rel = {
                    "subject": validate_term(rel["subject"], object_classes),
                    "predicate": validate_term(rel["predicate"], relationship_classes),
                    "object": validate_term(rel["object"], object_classes)
                }
                validated_graph["relationships"].append(validated_rel)
        
        print(f"Validated scene graph: {len(validated_graph['objects'])} objects, {len(validated_graph['relationships'])} relationships")
        return validated_graph
        
    except Exception as e:
        print(f"Vocabulary validation failed: {str(e)}")
        if strict:
            raise  # In strict mode, fail on validation errors
        return scene_graph  # In non-strict mode, return original

def get_gemini_response(prompt):
    """Get response from Gemini API and extract JSON scene graph"""
    try:
        # Initialize Gemini
        import google.generativeai as genai
        
        # Configure the model
        genai.configure(api_key='')
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Update prompt to force JSON-only response
        json_prompt = f"""
        {prompt}
        
        IMPORTANT: Respond ONLY with the JSON scene graph, no other text.
        The scene graph must be valid JSON with this exact structure:
        {{
            "objects": [
                {{"object": "object_name"}}
            ],
            "relationships": [
                {{"subject": "subject_name", "predicate": "predicate_name", "object": "object_name"}}
            ]
        }}
        """
        
        # Get response
        response = model.generate_content(json_prompt)
        response_text = response.text
        
        # Extract JSON if response contains markdown
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
            
        # Validate JSON structure
        scene_graph = json.loads(json_text)
        if not all(key in scene_graph for key in ["objects", "relationships"]):
            raise ValueError("Invalid scene graph structure")
            
        return json_text
        
    except Exception as e:
        print(f"Gemini API or JSON parsing failed: {str(e)}")
        print(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
        # Return minimal valid scene graph as fallback
        return json.dumps({
            "objects": [{"object": "person"}],
            "relationships": []
        })

def main():
    """Main execution function"""
    try:
        print("Starting scene graph prediction pipeline")
        
        # Setup
        model, processor, device = setup_environment()
        
        # Load data
        bbox_rel_data, object_classes, relationship_classes = load_annotations()
        
        # Process specific video
        video_id = "BPT87.mp4"
        scene_graphs = process_video_frames(video_id, bbox_rel_data)
        
        if not scene_graphs:
            print(f"No scene graphs found for video {video_id}")
            return
            
        # Split into train/test
        train_graphs, test_graphs = split_scene_graphs(scene_graphs)
        
        # Generate predictions
        vocabulary_type = 'closed'
        predictions = predict_scene_graphs(
            train_graphs,
            len(test_graphs),
            model,
            processor,
            device,
            vocabulary_type,
            object_classes,
            relationship_classes
        )
        
        # Visualize results
        visualize_results(test_graphs, predictions)
        
        print("Pipeline completed successfully")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()

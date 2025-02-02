# Import required libraries
import os
import google.generativeai as genai
import dotenv
import json
import logging
import traceback
import pickle
from glob import glob
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import base64

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment and API configurations"""
    try:
        logger.info("Starting environment setup...")
        dotenv.load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Environment setup completed")
        return model
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        raise

def load_annotations():
    """Load annotation files and object/relationship classes"""
    logger.info("Loading annotation files...")
    try:
        # Load object bbox and relationship data
        with open('data/object_bbox_and_relationship.pkl', 'rb') as f:
            bbox_rel_data = pickle.load(f)
        
        # Load object classes
        with open('data/object_classes.txt', 'r') as f:
            object_classes = f.read().splitlines()
            
        # Load relationship classes
        with open('data/relationship_classes.txt', 'r') as f:
            relationship_classes = f.read().splitlines()
            
        logger.info(f"Loaded annotations with {len(object_classes)} objects, {len(relationship_classes)} relationships")
        return bbox_rel_data, object_classes, relationship_classes
    
    except Exception as e:
        logger.error(f"Failed to load annotations: {str(e)}")
        raise

def process_video_frames(video_id, bbox_rel_data):
    """Process frames from a specific video folder"""
    try:
        logger.info(f"Processing video: {video_id}")
        
        # Get all frames for this video
        frame_path = f"./data/frames/{video_id}/*.png"
        frame_files = sorted(glob(frame_path))
        
        if not frame_files:
            logger.error(f"No frames found in {frame_path}")
            return None
            
        logger.info(f"Found {len(frame_files)} frames for video {video_id}")
        
        # Extract scene graphs
        scene_graphs = []
        for frame_file in tqdm(frame_files, desc="Extracting scene graphs"):
            # Construct the frame ID as it appears in the pkl file
            frame_name = os.path.basename(frame_file)
            frame_id = f"{video_id}/{frame_name}"
            
            # Get scene graph from annotations
            scene_graph = extract_scene_graph(frame_id, bbox_rel_data)
            
            if scene_graph:
                logger.debug(f"Found scene graph for frame {frame_id}")
                scene_graphs.append({
                    'frame_id': frame_id,
                    'scene_graph': scene_graph
                })
            else:
                logger.warning(f"No scene graph found for frame {frame_id}")
        
        logger.info(f"Extracted {len(scene_graphs)} scene graphs")
        return scene_graphs
    
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise

def extract_scene_graph(frame_id, bbox_rel_data):
    """Extract scene graph from annotations"""
    try:
        # Get annotations for this frame
        frame_anns = bbox_rel_data.get(frame_id, [])
        if not frame_anns:
            logger.debug(f"No annotations found for frame {frame_id}")
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
            logger.debug(f"Extracted scene graph with {len(scene_graph['objects'])} objects and {len(scene_graph['relationships'])} relationships")
            return scene_graph
        else:
            logger.debug(f"No valid objects found in annotations for frame {frame_id}")
            return None
    
    except Exception as e:
        logger.error(f"Failed to extract scene graph for frame {frame_id}: {str(e)}")
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
        
        logger.info(f"Split {len(scene_graphs)} graphs chronologically into {len(train_graphs)} train and {len(test_graphs)} test")
        logger.debug(f"Training frames: {train_graphs[0]['frame_id']} to {train_graphs[-1]['frame_id']}")
        logger.debug(f"Testing frames: {test_graphs[0]['frame_id']} to {test_graphs[-1]['frame_id']}")
        
        return train_graphs, test_graphs
    except Exception as e:
        logger.error(f"Failed to split scene graphs: {str(e)}")
        raise

def parse_model_response(response_text):
    """Parse model response to extract JSON"""
    try:
        # Remove markdown formatting if present
        if '```json' in response_text:
            json_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            json_text = response_text.split('```')[1].strip()
        else:
            json_text = response_text.strip()
            
        # Parse JSON
        scene_graph = json.loads(json_text)
        logger.debug(f"Successfully parsed response into scene graph with {len(scene_graph.get('objects', []))} objects")
        return scene_graph
    
    except Exception as e:
        logger.error(f"Failed to parse model response: {str(e)}")
        logger.debug(f"Response text: {response_text}")
        return None

def predict_scene_graphs(train_graphs, num_predictions, model, vocabulary_type='closed', 
                        object_classes=None, relationship_classes=None):
    """Generate scene graph predictions using frame images and scene graphs as context"""
    try:
        logger.info(f"Generating {num_predictions} sequential predictions using {vocabulary_type} vocabulary")
        predictions = []
        context_graphs = train_graphs.copy()
        
        # Debug initial context
        logger.debug("\n=======INITIAL CONTEXT========")
        logger.debug(f"Starting with {len(context_graphs)} training frames")
        logger.debug(f"Frame numbers: {[int(g['frame_id'].split('/')[-1].split('.')[0]) for g in context_graphs]}")
        logger.debug("============================\n")
        
        # Get test frame IDs to predict
        test_frame_nums = []
        last_train_num = int(context_graphs[-1]['frame_id'].split('/')[-1].split('.')[0])
        
        # Find existing frame files after the last training frame
        video_id = context_graphs[0]['frame_id'].split('/')[0]
        frame_pattern = f"./data/frames/{video_id}/*.png"
        all_frames = sorted(glob(frame_pattern))
        
        for frame_path in all_frames:
            frame_num = int(frame_path.split('/')[-1].split('.')[0])
            if frame_num > last_train_num:
                test_frame_nums.append(frame_num)
        
        logger.info(f"Found {len(test_frame_nums)} test frames to predict after frame {last_train_num}")
        
        # Prepare vocabulary context
        vocab_context = ""
        if vocabulary_type == 'closed':
            vocab_context = f"Available objects: {', '.join(object_classes)}\nAvailable relationships: {', '.join(relationship_classes)}"
        
        for i, next_frame_num in enumerate(test_frame_nums[:num_predictions], 1):
            logger.debug(f"\n=======PREDICTION {i}/{num_predictions}========")
            logger.debug(f"Predicting frame {next_frame_num:06d}")
            logger.debug(f"Context has {len(context_graphs)} frames")
            logger.debug(f"Last 3 context frames: {[int(g['frame_id'].split('/')[-1].split('.')[0]) for g in context_graphs[-3:]]}")
            
            # Load previous frame images as base64
            frame_contexts = []
            for ctx in context_graphs:
                frame_path = f"./data/frames/{ctx['frame_id']}"
                try:
                    with open(frame_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        frame_num = int(ctx['frame_id'].split('/')[-1].split('.')[0])
                        frame_contexts.append({
                            'frame_num': frame_num,
                            'image': img_data,
                            'scene_graph': ctx['scene_graph']
                        })
                except Exception as e:
                    logger.warning(f"Failed to load frame image {frame_path}: {e}")
            
            # Debug context information
            debug_context = "\n=======CONTEXT DETAILS========\n"
            for ctx in frame_contexts[-3:]:  # Show last 3 frames for brevity
                frame_num = ctx['frame_num']
                scene_graph = ctx['scene_graph']
                debug_context += f"\nFrame {frame_num:06d}:\n"
                debug_context += "Objects:\n"
                for obj in scene_graph['objects']:
                    debug_context += f"  - {obj['object']}\n"
                debug_context += "Relationships:\n"
                for rel in scene_graph['relationships']:
                    debug_context += f"  - {rel['subject']} {rel['predicate']} {rel['object']}\n"
            debug_context += "\n========================\n"
            
            logger.debug(debug_context)
            
            # Prepare context string with both images and scene graphs
            context_str = json.dumps([{
                'frame_num': ctx['frame_num'],
                'scene_graph': ctx['scene_graph']
            } for ctx in frame_contexts], indent=2)
            
            prompt = f"""Given the sequence of scene graphs below, predict the next scene graph for frame {next_frame_num:06d}.
            Consider the logical progression of objects, their interactions, and relationships.
            
            {vocab_context if vocabulary_type == 'closed' else ''}

            Previous frames (with scene graphs):
            {context_str}
            
            Predict the next scene graph that maintains temporal consistency with the sequence above.
            Return ONLY a JSON object with this structure:
            {{
                "objects": [
                    {{"object": "<object_name>", "attributes": []}},
                ],
                "relationships": [
                    {{"subject": "<object1>", "predicate": "<relationship>", "object": "<object2>"}}
                ]
            }}
            Do not include any other text or formatting, just the JSON object.
            """
            
            # Debug prompt
            logger.debug("\n=======PROMPT========\n")
            logger.debug(prompt)
            logger.debug("\n===================\n")
            
            response = model.generate_content(prompt)
            predicted_graph = parse_model_response(response.text)
            
            if predicted_graph:
                # Debug prediction and context update
                logger.debug("\n=======CONTEXT UPDATE========")
                logger.debug(f"Adding prediction for frame {next_frame_num:06d} to context")
                logger.debug("Predicted scene graph:")
                logger.debug("Objects:")
                for obj in predicted_graph['objects']:
                    logger.debug(f"  - {obj['object']}")
                logger.debug("Relationships:")
                for rel in predicted_graph['relationships']:
                    logger.debug(f"  - {rel['subject']} {rel['predicate']} {rel['object']}")
                
                # Create prediction entry with proper frame ID format
                prediction_entry = {
                    'frame_id': f"{video_id}/{next_frame_num:06d}.png",
                    'scene_graph': predicted_graph
                }
                
                predictions.append(prediction_entry)
                context_graphs.append(prediction_entry)
                
                logger.debug(f"Context size now: {len(context_graphs)} frames")
                logger.debug(f"Context frame numbers: {[int(g['frame_id'].split('/')[-1].split('.')[0]) for g in context_graphs[-3:]]}")
                logger.debug("============================\n")
            else:
                logger.warning(f"Failed to predict frame {next_frame_num:06d}")
                continue
                
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def visualize_scene_graph(scene_graph, title, ax):
    """Visualize scene graph using networkx"""
    try:
        if not scene_graph or 'objects' not in scene_graph:
            logger.error(f"Invalid scene graph structure for {title}: {scene_graph}")
            return
            
        G = nx.Graph()
        
        # Add object nodes
        for obj in scene_graph["objects"]:
            if isinstance(obj, dict) and "object" in obj:
                G.add_node(obj["object"], type="object")
            else:
                logger.warning(f"Invalid object structure: {obj}")
                continue
            
        # Add relationship edges
        for rel in scene_graph.get("relationships", []):
            if isinstance(rel, dict) and all(k in rel for k in ["subject", "predicate", "object"]):
                G.add_edge(rel["subject"], rel["object"], 
                          label=rel["predicate"])
            else:
                logger.warning(f"Invalid relationship structure: {rel}")
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
            logger.warning(f"No valid nodes to visualize for {title}")
            
        ax.set_title(title)
        ax.axis('off')
        
    except Exception as e:
        logger.error(f"Failed to visualize scene graph: {str(e)}")
        logger.debug(f"Scene graph: {scene_graph}")
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
        logger.error(f"Visualization failed: {str(e)}")
        raise

def visualize_results(test_graphs, predictions):
    """Visualize all results"""
    try:
        if not predictions:
            logger.warning("No predictions to visualize")
            return
            
        logger.info("Visualizing results...")
        for i, (test_graph, pred_graph) in enumerate(zip(test_graphs, predictions)):
            try:
                frame_path = f"./data/frames/{test_graph['frame_id']}"
                logger.debug(f"Visualizing comparison for frame {test_graph['frame_id']}")
                
                # Extract scene graph from prediction (it's nested in the prediction structure)
                pred_scene_graph = pred_graph['scene_graph'] if isinstance(pred_graph, dict) else pred_graph
                
                # Log the structures for debugging
                logger.debug(f"Ground truth structure: {test_graph['scene_graph'].keys()}")
                logger.debug(f"Prediction structure: {pred_scene_graph.keys()}")
                
                visualize_comparison(
                    frame_path,
                    test_graph['scene_graph'],
                    pred_scene_graph
                )
            except Exception as e:
                logger.error(f"Failed to visualize comparison {i+1}: {str(e)}")
                logger.debug(f"Ground truth: {test_graph}")
                logger.debug(f"Prediction: {pred_graph}")
                continue
                
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise

def validate_vocabulary(scene_graph, object_classes, relationship_classes):
    """Validate and normalize scene graph vocabulary against allowed classes"""
    try:
        validated_graph = {
            "objects": [],
            "relationships": []
        }
        
        # Helper function to find closest match in vocabulary
        def find_closest_match(word, vocab):
            # Direct match
            if word.lower() in [v.lower() for v in vocab]:
                return vocab[[v.lower() for v in vocab].index(word.lower())]
            
            # Common substitutions
            substitutions = {
                # Objects
                'person': ['human', 'man', 'woman', 'individual', 'someone'],
                'box': ['container', 'package', 'carton'],
                'door': ['entrance', 'doorway', 'gate'],
                'mirror': ['reflection', 'glass'],
                # Relationships
                'looking_at': ['watching', 'observing', 'viewing', 'gazing_at'],
                'holding': ['carrying', 'gripping', 'grasping'],
                'touching': ['contacting', 'in_contact_with'],
                'in_front_of': ['before', 'facing'],
                'behind': ['at_back_of', 'in_back_of'],
                'not_contacting': ['separate_from', 'away_from', 'not_touching'],
                'not_looking_at': ['looking_away', 'facing_away']
            }
            
            # Check substitutions
            for valid_term, alternatives in substitutions.items():
                if word.lower() in [alt.lower() for alt in alternatives] and valid_term in vocab:
                    return valid_term
            
            # Default fallbacks
            if vocab == relationship_classes:
                logger.warning(f"Unknown relationship '{word}', using 'other_relationship'")
                return 'other_relationship'
            else:
                logger.warning(f"Unknown object '{word}', using 'other_object'")
                return 'other_object'
        
        # Validate objects
        seen_objects = set()
        for obj in scene_graph.get("objects", []):
            if isinstance(obj, dict) and "object" in obj:
                obj_name = obj["object"]
                validated_name = find_closest_match(obj_name, object_classes)
                
                if validated_name not in seen_objects:
                    validated_graph["objects"].append({
                        "object": validated_name,
                        "attributes": obj.get("attributes", []),
                        "bbox": obj.get("bbox", [])
                    })
                    seen_objects.add(validated_name)
        
        # Validate relationships
        for rel in scene_graph.get("relationships", []):
            if isinstance(rel, dict) and all(k in rel for k in ["subject", "predicate", "object"]):
                validated_rel = {
                    "subject": find_closest_match(rel["subject"], object_classes),
                    "predicate": find_closest_match(rel["predicate"], relationship_classes),
                    "object": find_closest_match(rel["object"], object_classes)
                }
                validated_graph["relationships"].append(validated_rel)
        
        logger.debug(f"Validated scene graph: {len(validated_graph['objects'])} objects, {len(validated_graph['relationships'])} relationships")
        return validated_graph
        
    except Exception as e:
        logger.error(f"Vocabulary validation failed: {str(e)}")
        logger.debug(f"Original scene graph: {scene_graph}")
        return scene_graph  # Return original if validation fails

def main():
    """Main execution function"""
    try:
        logger.info("Starting scene graph prediction pipeline")
        
        # Setup
        model = setup_environment()
        
        # Load data
        bbox_rel_data, object_classes, relationship_classes = load_annotations()
        
        # Process specific video
        video_id = "BPT87.mp4"
        scene_graphs = process_video_frames(video_id, bbox_rel_data)
        
        if not scene_graphs:
            logger.warning(f"No scene graphs found for video {video_id}")
            return
            
        # Split into train/test
        train_graphs, test_graphs = split_scene_graphs(scene_graphs)
        
        # Generate predictions
        vocabulary_type = 'closed'  # or 'open'
        predictions = predict_scene_graphs(
            train_graphs,
            len(test_graphs),
            model,
            vocabulary_type,
            object_classes,
            relationship_classes
        )
        
        # Visualize results
        visualize_results(test_graphs, predictions)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()

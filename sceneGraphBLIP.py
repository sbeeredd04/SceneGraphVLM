import os
import logging
import json
import matplotlib.pyplot as plt

def setup_output_directory(video_id):
    """Create output directory structure in Google Drive"""
    try:
        output_dir = f"/content/drive/MyDrive/output/{video_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/predictions", exist_ok=True)
        
        # Setup logging file
        log_file = f"{output_dir}/debug.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        return output_dir
    except Exception as e:
        print(f"Failed to setup output directory: {str(e)}")
        raise

def predict_scene_graphs(train_graphs, num_predictions=3, model, processor, device,
                        vocabulary_type='closed', object_classes=None, relationship_classes=None):
    """Generate scene graph predictions using BLIP model with detailed prompting"""
    try:
        print(f"\n=== Starting BLIP Scene Graph Prediction ===")
        predictions = []
        context_graphs = train_graphs[-3:]  # Use last 3 frames for context
        
        # Get frame numbers for prediction
        last_train_num = int(context_graphs[-1]['frame_id'].split('/')[-1].split('.')[0])
        video_id = context_graphs[0]['frame_id'].split('/')[0]
        
        for i in range(num_predictions):
            next_frame_num = last_train_num + i + 1
            print(f"\n=== PREDICTION {i+1}/{num_predictions} ===")
            
            # Format detailed context
            context_str = "Based on the sequence of previous frames:\n\n"
            for ctx in context_graphs[-3:]:
                frame_num = ctx['frame_id'].split('/')[-1].split('.')[0]
                context_str += f"Frame {frame_num}:\n"
                context_str += "Objects present: " + ", ".join([obj['object'] for obj in ctx['scene_graph']['objects']]) + "\n"
                context_str += "Relationships:\n"
                for rel in ctx['scene_graph']['relationships']:
                    context_str += f"- {rel['subject']} {rel['predicate']} {rel['object']}\n"
                context_str += "\n"
            
            # Create detailed prompt
            prompt = f"""
            {context_str}
            
            Analyze this sequence and predict the next frame's scene graph.
            
            Available vocabulary:
            Objects: {', '.join(object_classes)}
            Relationships: {', '.join(relationship_classes)}
            
            Guidelines:
            1. Temporal Consistency:
               - Consider the flow of actions from previous frames
               - Maintain logical progression of movements
               
            2. Object Persistence:
               - Track existing objects across frames
               - Only remove objects if they likely left the scene
               
            3. Relationship Evolution:
               - Update spatial relationships based on movement
               - Consider changing interactions between objects
               
            4. Vocabulary Constraints:
               - Use only the provided object and relationship terms
               - Ensure all relationships use valid objects
            
            Describe the predicted scene graph in detail.
            """
            
            # Get BLIP prediction
            inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            blip_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Parse and validate prediction
            try:
                predicted_graph = parse_blip_response(blip_response, object_classes, relationship_classes)
                prediction_entry = {
                    'frame_id': f"{video_id}/{next_frame_num:06d}.png",
                    'scene_graph': predicted_graph
                }
                
                # Save prediction to file
                prediction_file = f"{output_dir}/predictions/frame_{next_frame_num:06d}.json"
                with open(prediction_file, 'w') as f:
                    json.dump(prediction_entry, f, indent=2)
                
                predictions.append(prediction_entry)
                context_graphs.append(prediction_entry)
                
            except Exception as e:
                print(f"Failed to process BLIP prediction: {str(e)}")
                continue
        
        return predictions
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        raise

def visualize_results(test_graphs, predictions, output_dir):
    """Visualize and save all results"""
    try:
        if not predictions:
            print("No predictions to visualize")
            return
            
        print("Visualizing results...")
        for i, (test_graph, pred_graph) in enumerate(zip(test_graphs, predictions)):
            try:
                frame_path = f"/content/drive/MyDrive/data/frames/{test_graph['frame_id']}"
                frame_num = test_graph['frame_id'].split('/')[-1].split('.')[0]
                
                # Create visualization
                fig = plt.figure(figsize=(15, 5))
                
                # Save visualization
                viz_path = f"{output_dir}/visualizations/comparison_{frame_num}.png"
                visualize_comparison(frame_path, test_graph['scene_graph'], pred_graph['scene_graph'])
                plt.savefig(viz_path)
                plt.close()
                
            except Exception as e:
                print(f"Failed to visualize comparison {i+1}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        raise

def main():
    """Main execution function"""
    try:
        print("Starting scene graph prediction pipeline")
        
        # Setup
        model, processor, device = setup_environment()
        
        # Load data
        bbox_rel_data, object_classes, relationship_classes = load_annotations()
        
        # Process video
        video_id = "BPT87.mp4"
        output_dir = setup_output_directory(video_id)
        
        scene_graphs = process_video_frames(video_id, bbox_rel_data)
        
        if not scene_graphs:
            print(f"No scene graphs found for video {video_id}")
            return
            
        # Split into train/test
        train_graphs, test_graphs = split_scene_graphs(scene_graphs)
        
        # Generate 3 predictions
        predictions = predict_scene_graphs(
            train_graphs,
            num_predictions=3,
            model=model,
            processor=processor,
            device=device,
            vocabulary_type='closed',
            object_classes=object_classes,
            relationship_classes=relationship_classes
        )
        
        # Visualize and save results
        visualize_results(test_graphs[:3], predictions, output_dir)
        
        print("Pipeline completed successfully")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import argparse
import yaml
import cv2
from src.pipeline.detector import DetectionPipeline
from src.utils.visualizer import HUDRenderer
from src.io.video_reader import ThreadedVideoReader
from src.io.video_writer import ThreadedVideoWriter

def main():
    parser = argparse.ArgumentParser(description='RoadSense AI v2 Inference Entry Point')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--config', default='configs/default.yaml', help='Config file')
    parser.add_argument('--show', action='store_true', help='Show window')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--no-slice', action='store_true', help='Disable slicing')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames to process (0 = all)')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.no_slice:
        config['inference']['use_slicing'] = False
    pipeline = DetectionPipeline(config)
    renderer = HUDRenderer(language=config.get('display', {}).get('language', 'en'))
    reader = ThreadedVideoReader(args.input).start()
    writer = None
    if args.save or config.get('output', {}).get('save_video', False):
        out_path = config.get('output', {}).get('output_path', 'runs/inference/output.mp4')
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = ThreadedVideoWriter(out_path, reader.width, reader.height, reader.fps).start()
    show = args.show or config.get('display', {}).get('show_window', False)
    hud_enabled = config.get('display', {}).get('hud_enabled', True)
    if show:
        cv2.namedWindow('RoadSense AI v2', cv2.WINDOW_NORMAL)
    print(f'Starting inference on {args.input}...')
    frames = 0
    import time
    t0 = time.time()
    while reader.more():
        frame = reader.read()
        if frame is None:
            continue
        detections, stats = pipeline.run(frame)
        out_frame = frame.copy()
        out_frame = renderer.draw_boxes(out_frame, detections)
        if hud_enabled:
            out_frame = renderer.draw_hud(out_frame, stats)
        if writer:
            writer.write(out_frame)
        if show:
            cv2.imshow('RoadSense AI v2', out_frame)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
        frames += 1
        print(f'\rProcessing frame {frames}...', end='')
        if args.max_frames > 0 and frames >= args.max_frames:
            print(f'\nReached max frames ({args.max_frames}). Stopping.')
            break
    reader.stop()
    if writer:
        writer.stop()
    cv2.destroyAllWindows()
    t1 = time.time()
    fps = frames / (t1 - t0) if t1 > t0 else 0
    print(f'Inference complete. Processed {frames} frames at {fps:.2f} FPS.')
if __name__ == '__main__':
    main()
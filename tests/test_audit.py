import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from detection_claim_audit.audit import audit, claims, evidence, matches
from detection_claim_audit.phrases import DETECTION, NODETECTION
from detection_claim_audit.__main__ import main

FIXTURES = Path(__file__).parent / 'fixtures'


class ClaimsTests(unittest.TestCase):
    def test_counts_and_sort(self):
        self.assertEqual(len(DETECTION), 48)
        self.assertEqual(len(NODETECTION), 32)
        for values in (DETECTION, NODETECTION):
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(list(values), sorted(values, key=len, reverse=True))

    def test_fences(self):
        for fence in ('```', '~~~', '   ````'):
            self.assertEqual(claims(fence+'\ndetects objects\n'+fence), (False, False))

    def test_inline(self):
        self.assertEqual(claims('`detects objects`'), (True, False))

    def test_normalization(self):
        self.assertEqual(claims("Doesn't detect objects in images"), (False, True))
        self.assertEqual(claims('REAL-TIME OBJECT\nDETECTION'), (True, False))
        self.assertEqual(claims('detects\n\nobjects'), (False, False))

    def test_longest_and_boundaries(self):
        self.assertEqual(matches('detects objects in images', reversed(DETECTION))[0], ['detects objects in images'])
        self.assertEqual(claims('predetect objectsx'), (False, False))


POSITIVES = [
'import ultralytics\nultralytics.YOLO("x")',
'import ultralytics as u\nu.YOLO("x")',
'from ultralytics import YOLO as Y\nY("x")',
'from ultralytics import YOLO\nwith YOLO("x") as m:\n m.predict(x)',
'from ultralytics import YOLO\nself.model=YOLO("x")\nself.model(x)',
'from ultralytics import YOLO\nYOLO("x").predict(x)',
'from ultralytics import YOLO\nm: object=YOLO("x")\nm.predict(x)',
'from ultralytics import YOLO\nm=YOLO("x")\nm(x)',
'import mediapipe as mp\nmp.tasks.vision.ObjectDetector.create_from_options(options)',
'from mediapipe.tasks.vision import ObjectDetector as O\nO.create_from_options(options)',
'from mediapipe.tasks.vision import ObjectDetector\nObjectDetector(options)',
'import torchvision as tv\ntv.models.detection.fasterrcnn_resnet50_fpn()',
'from torchvision.models.detection import ssd300_vgg16 as model\nmodel()',
'from detectron2.engine import DefaultPredictor as D\np=D(cfg)\np(image)',
'import detectron2.engine as e\ne.DefaultPredictor(cfg)',
'from transformers import pipeline as p\np("object-detection")',
'import transformers as t\nt.pipeline(task="object-detection")',
]
for method in ('detect','detect_for_video','detect_async'):
    POSITIVES.append('from mediapipe.tasks.vision import ObjectDetector\nm=ObjectDetector.create_from_options(o)\nm.'+method+'(x)')
NEGATIVES = [
'import ultralytics', 'import cv2\ncv2.imread("x")',
'from PIL import Image\nImage.open("x")', 'import cv2\ncv2.VideoCapture(0)',
'import cv2\ncv2.rectangle(image, a, b, c)',
'from torchvision.models import resnet50\nresnet50()',
'import face_recognition\nface_recognition.face_locations(x)',
'import mediapipe as mp\nmp.solutions.face_detection.FaceDetection()',
'import mediapipe as mp\nmp.solutions.hands.Hands()',
'import mediapipe as mp\nmp.solutions.pose.Pose()',
'import mediapipe as mp\nmp.tasks.vision.PoseLandmarker.create_from_options(o)',
'import mediapipe as mp\nmp.solutions.selfie_segmentation.SelfieSegmentation()',
'import mediapipe as mp\nmp.tasks.vision.ImageSegmenter.create_from_options(o)',
'model.track(x)', 'from deep_sort import DeepSORT\nDeepSORT()',
'import cv2\nn=cv2.dnn.readNet("x")\nn.forward()',
'import torch\ntorch.hub.load("ultralytics/yolov5", "yolov5s")',
'from ultralytics import YOLO\ndef YOLO(x): return x\nYOLO("x")',
'from mediapipe.tasks.vision import ObjectDetector\nclass ObjectDetector: pass\nObjectDetector()',
'from detectron2.engine import DefaultPredictor\ndef DefaultPredictor(x): return x\nDefaultPredictor(x)',
'from transformers import pipeline\ndef pipeline(x): return x\npipeline("object-detection")',
'from transformers import pipeline\npipeline("image-classification")',
'predict(x)\ndetect(x)', 'x="object detection"',
'import subprocess\nsubprocess.run(["yolo", "detect"])', 'not valid python !',
'from other import YOLO\nYOLO(x)',
'from ultralytics import YOLO\nYOLO=lambda x:x\nYOLO(x)',
]


class EvidenceTests(unittest.TestCase):
    def test_tracking_method_is_not_bound_evidence(self):
        from detection_claim_audit.audit import Evidence
        import ast
        v=Evidence(); v.bound['m']='yolo'
        v.visit(ast.parse('m.track(x)'))
        self.assertFalse(v.found)


class IntegrationTests(unittest.TestCase):
    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                status=main(list(map(str,args)))
            except SystemExit as e:
                status=e.code
        return status,out.getvalue(),err.getvalue()

    def test_fixtures(self):
        for fixture, status, severity, rule in [
            ('good',0,None,None), ('no_claim',0,None,None),
            ('bad',2,'HIGH','documented-detection-claim-false'),
            ('testonly',0,'MEDIUM','documented-detection-claim-false'),
            ('no_detection',2,'HIGH','documented-no-detection-claim-false')]:
            with self.subTest(fixture=fixture):
                code,out,_=self.cli(FIXTURES/fixture,'--json')
                self.assertEqual(code,status)
                result=json.loads(out)
                self.assertEqual(set(result),{'verdict','findings','files_scanned'})
                self.assertEqual(result['verdict'],'FAIL' if status==2 else 'PASS')
                self.assertEqual(len(result['findings']),int(severity is not None))
                if severity:
                    self.assertEqual(result['findings'][0]['severity'],severity)
                    self.assertEqual(result['findings'][0]['rule'],rule)
                    self.assertEqual(set(result['findings'][0]),{'rule','severity','message'})
        self.assertEqual(self.cli(FIXTURES/'testonly','--fail-on','MEDIUM')[0],2)

    def test_ignore_and_verbose(self):
        code,out,_=self.cli(FIXTURES/'bad','--ignore','README.md','--verbose')
        self.assertEqual(code,0)
        self.assertIn('Files scanned: 1',out)
        self.assertEqual(self.cli(FIXTURES/'good','--ignore','*.py')[0],2)

    def test_version_errors(self):
        self.assertEqual(self.cli('--version')[:2],(0,'0.1.0\n'))
        for args in [[],['--unknown'],['/nonexistent-root'],[str(FIXTURES/'bad'),'--fail-on','LOW']]:
            self.assertEqual(self.cli(*args)[0],1)

    def test_exclusions_and_docs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'README.md').write_text('detects objects')
            for component in ['docs','examples','fixtures']:
                folder=root/component;folder.mkdir();(folder/'source.py').write_text(POSITIVES[0])
            (root/'linked.py').symlink_to(root/'docs/source.py')
            (root/'linked_dir').symlink_to(root/'docs',target_is_directory=True)
            self.assertEqual(audit(root)['findings'][0]['severity'],'HIGH')
            (root/'README.md').write_text('Nothing promised.')
            (root/'docs/note.md').write_text('detects objects')
            self.assertEqual(audit(root)['verdict'],'FAIL')

    def test_self_audit(self):
        self.assertEqual(audit(Path(__file__).resolve().parents[1])['verdict'],'PASS')


def add_test(cls, name, function, value, expected):
    def test(self):
        self.assertEqual(function(value), expected)
    setattr(cls, name, test)


for i,phrase in enumerate(DETECTION):
    add_test(ClaimsTests,'test_positive_%02d'%i,claims,phrase,(True,False))
for i,phrase in enumerate(NODETECTION):
    add_test(ClaimsTests,'test_negative_%02d'%i,claims,phrase,(False,True))
for i,source in enumerate(POSITIVES):
    add_test(EvidenceTests,'test_provider_%02d'%i,evidence,source,True)
for i,source in enumerate(NEGATIVES):
    add_test(EvidenceTests,'test_nonprovider_%02d'%i,evidence,source,False)

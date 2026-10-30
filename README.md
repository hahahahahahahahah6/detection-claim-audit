# Box-level assertion auditor

An offline static auditor comparing documented box-level finding promises with
Python provider evidence. Version 0.1.0; Python 3.9 or newer; no runtime dependencies.
Source is parsed with AST and never executed.

```sh
pip install detection-claim-audit
 detection-claim-audit ./project --json
python3 -m detection_claim_audit ./project --verbose
 detection-claim-audit ./project --ignore 'vendor/*' --fail-on MEDIUM
 detection-claim-audit --version
```

The root README.md and Markdown files beneath any docs path component supply
assertions. Backtick and tilde fences are excluded; inline code counts.
Blank lines separate paragraphs; wrapped lines join. Case, hyphens, whitespace,
and contracted negatives are normalized. Exact templates use word boundaries,
with longer templates first and negative spans removed before positive matching.
The complete template tuples are in phrases.py.

Python files supply evidence, except symlinks and files beneath fixtures, docs,
or examples. Files beneath tests or named test_*.py or *_test.py supply only
test evidence. Invalid syntax supplies none. Repeat --ignore to exclude relative
POSIX paths using fnmatch globs. The scanned count includes eligible documents
and source files after exclusions.

Supported providers are ultralytics YOLO, mediapipe tasks ObjectDetector,
torchvision's nine listed box-model constructors, detectron2 DefaultPredictor,
and transformers pipeline with the literal task shown below. Imports and aliases
are resolved within each module; local definitions shadow imported names.
Construction counts; supported bound calls, annotated assignments, attribute
assignments, context-manager targets, and chained constructions are recognized.
YOLO predict and direct calls, mediapipe detect, detect_for_video, detect_async,
and DefaultPredictor direct calls are supported.

```text
object-detection
fasterrcnn_resnet50_fpn
fasterrcnn_resnet50_fpn_v2
fasterrcnn_mobilenet_v3_large_fpn
fasterrcnn_mobilenet_v3_large_320_fpn
retinanet_resnet50_fpn
retinanet_resnet50_fpn_v2
ssd300_vgg16
ssdlite320_mobilenet_v3_large
fcos_resnet50_fpn
```

At most one finding is emitted for each assertion family:

```text
documented-detection-claim-false: HIGH without production evidence;
  MEDIUM if evidence exists only in tests.
documented-no-detection-claim-false: HIGH when production evidence contradicts
  a negative assertion.
```

Without an assertion the result is PASS. HIGH fails by default; --fail-on MEDIUM
also fails on MEDIUM. JSON contains verdict, findings (rule, severity, message),
and files_scanned. Human output includes the verdict and each finding;
--verbose adds the count. Exit codes: 0 for PASS, 2 for FAIL, 1 for invalid or
unreadable roots and argument errors. --version prints 0.1.0.

Limitations: this checks syntax-level fingerprints, not runtime success, model
weights, accuracy, or whether a constructed model is ever used. Bare imports,
image loading, camera capture, supplied-box drawing, whole-image classification,
face/hand/pose processing, pixel segmentation, tracking, strings, shell commands,
and subprocess helpers supply no evidence. Generic cv2.dnn readNet/forward and
torch.hub loaded models are intentionally ambiguous and unsupported. Dynamic
imports, intermodule bindings, and runtime task strings are unsupported.

```sh
python3 -m unittest discover
python3 -m build
```

The source distribution includes the tests and fixtures. Licensed under MIT.

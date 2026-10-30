"""Offline documentation assertions and provider-resolved Python evidence."""
import ast
import fnmatch
import os
from pathlib import Path
import re
from .phrases import DETECTION, NODETECTION

MODELS = {'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
          'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn',
          'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 'ssd300_vgg16',
          'ssdlite320_mobilenet_v3_large', 'fcos_resnet50_fpn'}


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        if fence:
            if re.match(r'^ {0,3}' + re.escape(fence[0]) + '{' + str(fence[1]) + r',}\s*$', line):
                fence = None
            continue
        match = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if match:
            fence = (match[1][0], len(match[1]))
            lines.append('')
        else:
            lines.append(line)
    return '\n'.join(lines)


def matches(text, phrases):
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text):
            found.append(phrase)
            text = re.sub(pattern, ' ', text)
    return found, text


def claims(text):
    positive = negative = False
    for paragraph in re.split(r'\n\s*\n', prose(text)):
        negatives, remainder = matches(normalize(paragraph), NODETECTION)
        positives, _ = matches(remainder, DETECTION)
        positive |= bool(positives)
        negative |= bool(negatives)
    return positive, negative


class Evidence(ast.NodeVisitor):
    def __init__(self):
        self.aliases = {}
        self.bound = {}
        self.found = False

    def path(self, node):
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = self.path(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    def key(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self.key(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    def kind(self, node):
        if not isinstance(node, ast.Call):
            return self.bound.get(self.key(node))
        path = self.path(node.func)
        if path.startswith('ultralytics.') and path.endswith('.YOLO'):
            return 'yolo'
        if path.startswith('mediapipe.') and (path.endswith('.ObjectDetector') or path.endswith('.ObjectDetector.create_from_options')):
            return 'mp'
        if path.startswith('detectron2.') and path.endswith('.DefaultPredictor'):
            return 'd2'
        if path.startswith('torchvision.models.detection.') and path.split('.')[-1] in MODELS:
            return 'tv'
        if path == 'transformers.pipeline':
            args = list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == 'task']
            if any(isinstance(a, ast.Constant) and a.value == 'object-detection' for a in args):
                return 'pipeline'
        return None

    def visit_Import(self, node):
        for item in node.names:
            name = item.asname or item.name.split('.')[0]
            self.aliases[name] = item.name if item.asname else name
            self.bound.pop(name, None)

    def visit_ImportFrom(self, node):
        for item in node.names:
            name = item.asname or item.name
            self.aliases[name] = (node.module + '.' + item.name) if node.module and not node.level else ''
            self.bound.pop(name, None)

    def bind(self, target, value):
        key = self.key(target)
        if key:
            kind = self.kind(value)
            self.bound.pop(key, None)
            self.aliases.pop(key, None)
            if kind:
                self.bound[key] = kind
        elif isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                self.bind(child, None)

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self.bind(target, node.value)

    def visit_AnnAssign(self, node):
        if node.value:
            self.visit(node.value)
        self.bind(node.target, node.value)

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.bind(item.optional_vars, item.context_expr)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_Call(self, node):
        if self.kind(node):
            self.found = True
        kind = self.kind(node.func)
        if kind in {'yolo', 'd2'}:
            self.found = True
        if isinstance(node.func, ast.Attribute):
            kind = self.kind(node.func.value)
            if (kind == 'yolo' and node.func.attr == 'predict') or (kind == 'mp' and node.func.attr in {'detect', 'detect_for_video', 'detect_async'}):
                self.found = True
        self.generic_visit(node)

    def definition(self, node):
        self.aliases.pop(node.name, None)
        self.bound.pop(node.name, None)
        # Definitions shadow imported providers throughout this module. Function
        # bodies share import resolution but their local bindings do not leak.
        aliases, bound = self.aliases.copy(), self.bound.copy()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                self.aliases.pop(arg.arg, None)
                self.bound.pop(arg.arg, None)
        for statement in node.body:
            self.visit(statement)
        self.aliases, self.bound = aliases, bound

    visit_FunctionDef = definition
    visit_AsyncFunctionDef = definition
    visit_ClassDef = definition


def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    visitor = Evidence()
    visitor.visit(tree)
    return visitor.found


def audit(root, ignore=(), fail_on='HIGH'):
    root = Path(root)
    if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
        raise ValueError('Root must be a readable directory')
    positive = negative = production = test = False
    count = 0
    def walk_error(error):
        raise ValueError(str(error))
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink())
        for filename in sorted(files):
            path = Path(directory, filename)
            rel = path.relative_to(root)
            if path.is_symlink() or any(fnmatch.fnmatchcase(rel.as_posix(), pattern) for pattern in ignore):
                continue
            doc = rel.as_posix() == 'README.md' or (path.suffix == '.md' and 'docs' in rel.parts[:-1])
            source = path.suffix == '.py' and not set(rel.parts[:-1]) & {'fixtures', 'docs', 'examples'}
            if not (doc or source):
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as error:
                raise ValueError(str(error)) from error
            count += 1
            if doc:
                pos, neg = claims(content)
                positive |= pos
                negative |= neg
            if source and evidence(content):
                is_test = 'tests' in rel.parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py')
                test |= is_test
                production |= not is_test
    findings = []
    if positive and not production:
        findings.append(dict(rule='documented-detection-claim-false', severity='MEDIUM' if test else 'HIGH', message='Positive assertion lacks production provider evidence.'))
    if negative and production:
        findings.append(dict(rule='documented-no-detection-claim-false', severity='HIGH', message='Negative assertion conflicts with production provider evidence.'))
    failed = any(f['severity'] == 'HIGH' or fail_on == 'MEDIUM' for f in findings)
    return dict(verdict='FAIL' if failed else 'PASS', findings=findings, files_scanned=count)

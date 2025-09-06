#!/usr/bin/env python3
"""
Code Quality Analyzer

Analyzes code quality metrics for the Genetic Algorithm project including:
- Line counts and code complexity
- Documentation coverage
- Test coverage estimation
- Code organization metrics
- Dependency analysis
"""

import os
import sys
import ast
import re
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter


class CodeQualityAnalyzer:
    """Analyzes code quality metrics for the GA project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.metrics = {
            'files': {},
            'modules': {},
            'overall': {},
            'documentation': {},
            'testing': {},
            'dependencies': {}
        }
    
    def analyze_all(self) -> Dict[str, Any]:
        """Run complete code quality analysis."""
        print("Analyzing code quality...")
        
        # Find all Python files
        python_files = self._find_python_files()
        print(f"Found {len(python_files)} Python files")
        
        # Analyze each file
        for file_path in python_files:
            self._analyze_file(file_path)
        
        # Calculate overall metrics
        self._calculate_overall_metrics()
        self._analyze_documentation_coverage()
        self._analyze_test_coverage()
        self._analyze_dependencies()
        
        return self.metrics
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        
        # Core directories to analyze
        core_dirs = [
            self.project_root,
            self.project_root / "ga_components",
            self.project_root / "Compressors",
            self.project_root / "tests"
        ]
        
        for directory in core_dirs:
            if directory.exists():
                python_files.extend(directory.glob("*.py"))
        
        # Filter out __pycache__ and other generated files
        return [f for f in python_files if "__pycache__" not in str(f)]
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Calculate metrics
            metrics = {
                'path': str(file_path.relative_to(self.project_root)),
                'lines_total': len(content.splitlines()),
                'lines_code': self._count_code_lines(content),
                'lines_comments': self._count_comment_lines(content),
                'lines_docstrings': self._count_docstring_lines(tree),
                'lines_blank': content.count('\n\n'),
                'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'complexity': self._calculate_complexity(tree),
                'has_main_docstring': self._has_module_docstring(tree),
                'documented_functions': self._count_documented_functions(tree),
                'documented_classes': self._count_documented_classes(tree)
            }
            
            # Categorize file
            category = self._categorize_file(file_path)
            metrics['category'] = category
            
            self.metrics['files'][str(file_path.relative_to(self.project_root))] = metrics
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def _count_code_lines(self, content: str) -> int:
        """Count lines of actual code (excluding comments and blank lines)."""
        lines = content.splitlines()
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                code_lines += 1
        
        return code_lines
    
    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines."""
        lines = content.splitlines()
        return sum(1 for line in lines if line.strip().startswith('#'))
    
    def _count_docstring_lines(self, tree: ast.AST) -> int:
        """Count lines in docstrings."""
        docstring_lines = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstring_lines += len(docstring.splitlines())
        
        return docstring_lines
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _has_module_docstring(self, tree: ast.AST) -> bool:
        """Check if module has a docstring."""
        return ast.get_docstring(tree) is not None
    
    def _count_documented_functions(self, tree: ast.AST) -> Tuple[int, int]:
        """Count documented vs total functions."""
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        documented = sum(1 for func in functions if ast.get_docstring(func))
        return documented, len(functions)
    
    def _count_documented_classes(self, tree: ast.AST) -> Tuple[int, int]:
        """Count documented vs total classes."""
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        documented = sum(1 for cls in classes if ast.get_docstring(cls))
        return documented, len(classes)
    
    def _categorize_file(self, file_path: Path) -> str:
        """Categorize file by its purpose."""
        path_str = str(file_path)
        
        if "test" in path_str.lower():
            return "test"
        elif "ga_components" in path_str:
            return "component"
        elif "Compressors" in path_str:
            return "compressor"
        elif file_path.name in ["main.py", "genetic_algorithm.py"]:
            return "core"
        elif file_path.name in ["ga_config.py", "ga_logging.py", "ga_exceptions.py"]:
            return "utility"
        else:
            return "other"
    
    def _calculate_overall_metrics(self) -> None:
        """Calculate overall project metrics."""
        files = self.metrics['files']
        
        # Aggregate metrics
        total_lines = sum(f['lines_total'] for f in files.values())
        total_code = sum(f['lines_code'] for f in files.values())
        total_comments = sum(f['lines_comments'] for f in files.values())
        total_docstrings = sum(f['lines_docstrings'] for f in files.values())
        total_functions = sum(f['functions'] for f in files.values())
        total_classes = sum(f['classes'] for f in files.values())
        
        # Calculate ratios
        comment_ratio = (total_comments / total_lines) * 100 if total_lines > 0 else 0
        docstring_ratio = (total_docstrings / total_lines) * 100 if total_lines > 0 else 0
        code_ratio = (total_code / total_lines) * 100 if total_lines > 0 else 0
        
        self.metrics['overall'] = {
            'total_files': len(files),
            'total_lines': total_lines,
            'total_code_lines': total_code,
            'total_comment_lines': total_comments,
            'total_docstring_lines': total_docstrings,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'comment_ratio': round(comment_ratio, 2),
            'docstring_ratio': round(docstring_ratio, 2),
            'code_ratio': round(code_ratio, 2),
            'avg_file_size': round(total_lines / len(files), 1) if files else 0,
            'avg_complexity': round(sum(f['complexity'] for f in files.values()) / len(files), 1) if files else 0
        }
        
        # Category breakdown
        categories = defaultdict(list)
        for file_data in files.values():
            categories[file_data['category']].append(file_data)
        
        self.metrics['modules'] = {}
        for category, file_list in categories.items():
            self.metrics['modules'][category] = {
                'file_count': len(file_list),
                'total_lines': sum(f['lines_total'] for f in file_list),
                'avg_lines': round(sum(f['lines_total'] for f in file_list) / len(file_list), 1),
                'total_functions': sum(f['functions'] for f in file_list),
                'total_classes': sum(f['classes'] for f in file_list)
            }
    
    def _analyze_documentation_coverage(self) -> None:
        """Analyze documentation coverage."""
        files = self.metrics['files']
        
        # Module docstring coverage
        modules_with_docstrings = sum(1 for f in files.values() if f['has_main_docstring'])
        module_docstring_coverage = (modules_with_docstrings / len(files)) * 100 if files else 0
        
        # Function docstring coverage
        total_functions = sum(f['functions'] for f in files.values())
        documented_functions = sum(f['documented_functions'][0] for f in files.values())
        function_docstring_coverage = (documented_functions / total_functions) * 100 if total_functions > 0 else 0
        
        # Class docstring coverage
        total_classes = sum(f['classes'] for f in files.values())
        documented_classes = sum(f['documented_classes'][0] for f in files.values())
        class_docstring_coverage = (documented_classes / total_classes) * 100 if total_classes > 0 else 0
        
        self.metrics['documentation'] = {
            'module_docstring_coverage': round(module_docstring_coverage, 1),
            'function_docstring_coverage': round(function_docstring_coverage, 1),
            'class_docstring_coverage': round(class_docstring_coverage, 1),
            'overall_documentation_ratio': round(self.metrics['overall']['docstring_ratio'], 1),
            'files_needing_docs': [
                f['path'] for f in files.values() 
                if not f['has_main_docstring'] or f['documented_functions'][1] > f['documented_functions'][0]
            ]
        }
    
    def _analyze_test_coverage(self) -> None:
        """Analyze test coverage based on file structure."""
        files = self.metrics['files']
        
        # Categorize files
        test_files = [f for f in files.values() if f['category'] == 'test']
        non_test_files = [f for f in files.values() if f['category'] != 'test']
        
        # Test metrics
        test_lines = sum(f['lines_code'] for f in test_files)
        code_lines = sum(f['lines_code'] for f in non_test_files)
        test_ratio = (test_lines / code_lines) * 100 if code_lines > 0 else 0
        
        # Identify components that might need tests
        component_files = [f['path'] for f in files.values() if f['category'] == 'component']
        test_file_names = [f['path'] for f in test_files]
        
        # Simple heuristic: check if component has corresponding test
        components_with_tests = []
        components_without_tests = []
        
        for comp_file in component_files:
            comp_name = Path(comp_file).stem
            has_test = any(comp_name in test_file for test_file in test_file_names)
            if has_test:
                components_with_tests.append(comp_file)
            else:
                components_without_tests.append(comp_file)
        
        self.metrics['testing'] = {
            'test_file_count': len(test_files),
            'test_lines_of_code': test_lines,
            'test_to_code_ratio': round(test_ratio, 1),
            'components_with_tests': len(components_with_tests),
            'components_without_tests': len(components_without_tests),
            'testing_coverage_estimate': round((len(components_with_tests) / len(component_files)) * 100, 1) if component_files else 0,
            'files_needing_tests': components_without_tests
        }
    
    def _analyze_dependencies(self) -> None:
        """Analyze project dependencies."""
        files = self.metrics['files']
        
        # Count internal imports (within project)
        internal_imports = Counter()
        external_imports = Counter()
        
        for file_path, file_data in files.items():
            try:
                with open(self.project_root / file_path, 'r') as f:
                    content = f.read()
                
                # Find import statements
                import_pattern = r'from\s+([\w.]+)\s+import|import\s+([\w.]+)'
                imports = re.findall(import_pattern, content)
                
                for from_import, direct_import in imports:
                    module = from_import or direct_import
                    module = module.split('.')[0]  # Get base module
                    
                    if module in ['ga_components', 'Compressors'] or module.startswith('ga_'):
                        internal_imports[module] += 1
                    elif module not in ['os', 'sys', 'json', 'time', 'typing', 'pathlib']:
                        external_imports[module] += 1
                        
            except Exception:
                pass
        
        self.metrics['dependencies'] = {
            'internal_dependencies': dict(internal_imports.most_common(10)),
            'external_dependencies': dict(external_imports.most_common(10)),
            'most_imported_internal': internal_imports.most_common(1)[0] if internal_imports else None,
            'dependency_count': len(external_imports)
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive quality report."""
        metrics = self.metrics
        
        report = []
        report.append("# Code Quality Report\n")
        report.append("Generated automatically by Code Quality Analyzer\n")
        
        # Overall metrics
        overall = metrics['overall']
        report.append("## Overall Statistics\n")
        report.append(f"- **Total Files**: {overall['total_files']}")
        report.append(f"- **Total Lines**: {overall['total_lines']:,}")
        report.append(f"- **Code Lines**: {overall['total_code_lines']:,} ({overall['code_ratio']:.1f}%)")
        report.append(f"- **Comment Lines**: {overall['total_comment_lines']:,} ({overall['comment_ratio']:.1f}%)")
        report.append(f"- **Documentation Lines**: {overall['total_docstring_lines']:,} ({overall['docstring_ratio']:.1f}%)")
        report.append(f"- **Functions**: {overall['total_functions']}")
        report.append(f"- **Classes**: {overall['total_classes']}")
        report.append(f"- **Average File Size**: {overall['avg_file_size']} lines")
        report.append(f"- **Average Complexity**: {overall['avg_complexity']}")
        report.append("")
        
        # Module breakdown
        report.append("## Module Breakdown\n")
        modules = metrics['modules']
        
        for category, data in modules.items():
            report.append(f"### {category.title()} Files")
            report.append(f"- Files: {data['file_count']}")
            report.append(f"- Total Lines: {data['total_lines']:,}")
            report.append(f"- Average Size: {data['avg_lines']} lines")
            report.append(f"- Functions: {data['total_functions']}")
            report.append(f"- Classes: {data['total_classes']}")
            report.append("")
        
        # Documentation coverage
        doc = metrics['documentation']
        report.append("## Documentation Coverage\n")
        report.append(f"- **Module Docstrings**: {doc['module_docstring_coverage']:.1f}%")
        report.append(f"- **Function Docstrings**: {doc['function_docstring_coverage']:.1f}%")
        report.append(f"- **Class Docstrings**: {doc['class_docstring_coverage']:.1f}%")
        
        if doc['files_needing_docs']:
            report.append(f"\n**Files needing documentation** ({len(doc['files_needing_docs'])}):")
            for file_path in doc['files_needing_docs'][:5]:  # Show first 5
                report.append(f"- {file_path}")
            if len(doc['files_needing_docs']) > 5:
                report.append(f"- ... and {len(doc['files_needing_docs']) - 5} more")
        report.append("")
        
        # Testing coverage
        test = metrics['testing']
        report.append("## Testing Coverage\n")
        report.append(f"- **Test Files**: {test['test_file_count']}")
        report.append(f"- **Test Lines**: {test['test_lines_of_code']:,}")
        report.append(f"- **Test to Code Ratio**: {test['test_to_code_ratio']:.1f}%")
        report.append(f"- **Component Test Coverage**: {test['testing_coverage_estimate']:.1f}%")
        
        if test['files_needing_tests']:
            report.append(f"\n**Components needing tests** ({len(test['files_needing_tests'])}):")
            for file_path in test['files_needing_tests']:
                report.append(f"- {file_path}")
        report.append("")
        
        # Dependencies
        deps = metrics['dependencies']
        report.append("## Dependencies\n")
        report.append(f"- **External Dependencies**: {deps['dependency_count']}")
        
        if deps['external_dependencies']:
            report.append("\n**Top External Dependencies:**")
            for module, count in list(deps['external_dependencies'].items())[:5]:
                report.append(f"- {module}: {count} imports")
        
        if deps['internal_dependencies']:
            report.append("\n**Top Internal Dependencies:**")
            for module, count in list(deps['internal_dependencies'].items())[:5]:
                report.append(f"- {module}: {count} imports")
        report.append("")
        
        # Quality grades
        report.append("## Quality Grades\n")
        
        # Documentation grade
        doc_score = (doc['module_docstring_coverage'] + doc['function_docstring_coverage'] + doc['class_docstring_coverage']) / 3
        doc_grade = self._calculate_grade(doc_score)
        report.append(f"- **Documentation**: {doc_grade} ({doc_score:.1f}%)")
        
        # Test coverage grade  
        test_grade = self._calculate_grade(test['testing_coverage_estimate'])
        report.append(f"- **Testing**: {test_grade} ({test['testing_coverage_estimate']:.1f}%)")
        
        # Code organization grade (based on comment ratio and structure)
        org_score = min(100, overall['comment_ratio'] * 2 + overall['docstring_ratio'])
        org_grade = self._calculate_grade(org_score)
        report.append(f"- **Code Organization**: {org_grade} ({org_score:.1f}%)")
        
        # Overall grade
        overall_score = (doc_score + test['testing_coverage_estimate'] + org_score) / 3
        overall_grade = self._calculate_grade(overall_score)
        report.append(f"- **Overall**: {overall_grade} ({overall_score:.1f}%)")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations\n")
        
        if doc_score < 70:
            report.append("- **Improve Documentation**: Add docstrings to modules, functions, and classes")
        if test['testing_coverage_estimate'] < 80:
            report.append("- **Increase Test Coverage**: Add tests for components without test coverage")
        if overall['comment_ratio'] < 10:
            report.append("- **Add Comments**: Increase inline comments for complex code sections")
        if overall['avg_complexity'] > 15:
            report.append("- **Reduce Complexity**: Break down complex functions into smaller units")
        
        report.append("- **Continue Documentation**: The project shows good documentation practices - maintain this standard")
        report.append("- **Expand Testing**: Consider adding more edge case and integration tests")
        
        return "\n".join(report)
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from percentage score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


def main():
    """Main entry point for code quality analysis."""
    # Get project root (assume script is in tools/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Analyzing project: {project_root}")
    
    # Run analysis
    analyzer = CodeQualityAnalyzer(str(project_root))
    metrics = analyzer.analyze_all()
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    report_path = project_root / "docs" / "CODE_QUALITY_REPORT.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Quality report saved to: {report_path}")
    
    # Save raw metrics
    metrics_path = project_root / "docs" / "quality_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Raw metrics saved to: {metrics_path}")
    
    # Print summary
    overall = metrics['overall']
    doc = metrics['documentation']
    test = metrics['testing']
    
    print("\nQuality Summary:")
    print(f"  Total Lines: {overall['total_lines']:,}")
    print(f"  Documentation: {doc['function_docstring_coverage']:.1f}%")
    print(f"  Test Coverage: {test['testing_coverage_estimate']:.1f}%")
    print(f"  Overall Grade: {analyzer._calculate_grade((doc['function_docstring_coverage'] + test['testing_coverage_estimate']) / 2)}")


if __name__ == "__main__":
    main()
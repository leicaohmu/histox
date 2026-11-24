"""Tests for histox.registry and the registries built on it."""

import unittest

from histox.registry import (DuplicateEntryError, Registry, RegistryError,
                             UnknownEntryError, get_registry, list_registries)


class _Unknown(RegistryError, KeyError):
    pass


class _Duplicate(RegistryError):
    pass


class RegistryTest(unittest.TestCase):

    def setUp(self):
        self._made = []

    def tearDown(self):
        from histox import registry as reg
        for kind in self._made:
            reg._REGISTRIES.pop(kind, None)

    def _registry(self, kind, **kwargs):
        r = Registry(kind, **kwargs)
        self._made.append(kind)
        return r

    def test_decorator_forms(self):
        r = self._registry('test.forms')

        @r.register
        def alpha():
            return 'a'

        @r.register('beta', license='MIT')
        def build_beta():
            return 'b'

        r.add('gamma', lambda: 'c')
        self.assertEqual(r.names(), ['alpha', 'beta', 'gamma'])
        self.assertIs(r['alpha'], alpha)
        self.assertEqual(r.get('beta')(), 'b')
        self.assertEqual(dict(r.metadata('beta')), {'license': 'MIT'})
        self.assertEqual(dict(r.metadata('alpha')), {})

    def test_mapping_interface(self):
        r = self._registry('test.mapping')
        r.add('x', 1)
        r.add('y', 2)
        self.assertIn('x', r)
        self.assertNotIn('z', r)
        self.assertNotIn(3, r)
        self.assertEqual(len(r), 2)
        self.assertEqual(sorted(r.keys()), ['x', 'y'])
        self.assertEqual(dict(r.items()), {'x': 1, 'y': 2})

    def test_duplicates_error_by_default(self):
        r = self._registry('test.dup_error')
        r.add('x', 1)
        with self.assertRaises(DuplicateEntryError):
            r.add('x', 2)
        r.add('x', 3, replace=True)
        self.assertEqual(r['x'], 3)

    def test_duplicates_replace_policy(self):
        r = self._registry('test.dup_replace', duplicates='replace')
        r.add('x', 1)
        r.add('x', 2)
        self.assertEqual(r['x'], 2)
        with self.assertRaises(DuplicateEntryError):
            r.add('x', 3, replace=False)

    def test_unknown_name_lists_available(self):
        r = self._registry('test.unknown', label='widget')
        r.add('x', 1)
        with self.assertRaises(UnknownEntryError) as ctx:
            r.get('nope')
        self.assertIsInstance(ctx.exception, KeyError)
        self.assertIn("Unknown widget 'nope'", str(ctx.exception))
        self.assertIn("['x']", str(ctx.exception))
        self.assertIsNone(r.get('nope', None))

    def test_extras_hint(self):
        r = self._registry('test.extras', extras={'big_model': 'histox-contrib'})
        self.assertEqual(r.extras_package('big_model'), 'histox-contrib')
        with self.assertRaises(UnknownEntryError) as ctx:
            r.get('big_model')
        self.assertIn('pip install histox-contrib', str(ctx.exception))
        r.add('big_model', object())
        self.assertIn('big_model', r)

    def test_normalize_and_custom_errors(self):
        r = self._registry('test.norm', normalize=lambda s: s.strip().lower(),
                           unknown_error=_Unknown, duplicate_error=_Duplicate)
        r.add(' Survival ', 1)
        self.assertEqual(r.names(), ['survival'])
        self.assertEqual(r['SURVIVAL'], 1)
        with self.assertRaises(_Duplicate):
            r.add('survival', 2)
        with self.assertRaises(_Unknown):
            r['other']

    def test_invalid_names(self):
        r = self._registry('test.invalid')
        with self.assertRaises(ValueError):
            r.add('  ', 1)
        with self.assertRaises(ValueError):
            r.add(None, 1)  # object without __name__
        self.assertNotIn('', r)

    def test_unregister(self):
        r = self._registry('test.unregister')
        r.add('x', 1)
        r.unregister('x')
        self.assertNotIn('x', r)
        with self.assertRaises(UnknownEntryError):
            r.unregister('x')

    def test_global_index(self):
        r = self._registry('test.index')
        self.assertIs(get_registry('test.index'), r)
        self.assertIn('test.index', list_registries())
        with self.assertRaises(DuplicateEntryError):
            Registry('test.index')
        with self.assertRaises(UnknownEntryError):
            get_registry('test.no_such_registry')


class MigratedRegistriesTest(unittest.TestCase):
    """The existing registries are Registry instances with unchanged APIs."""

    def test_all_registries_exist(self):
        import histox  # noqa: F401
        import histox.mil  # noqa: F401  (MIL registries are created on import)
        for kind in ('extractor.torch', 'extractor.tensorflow', 'mil_model',
                     'mil_trainer', 'task', 'method'):
            self.assertIsInstance(get_registry(kind), Registry)

    def test_extractor_registry(self):
        import histox as hx
        from histox.model.extractors import _registry as ext
        names = hx.model.list_extractors()
        self.assertEqual(names, sorted(names))
        self.assertTrue(hx.model.is_extractor('resnet50'))   # '_imagenet' alias
        self.assertEqual(ext._torch_extractors.extras_package('gigapath'),
                         'slideflow-noncommercial')

        @ext.register_torch('registry_test_extractor', license='MIT')
        def registry_test_extractor(**kwargs):
            return 'built'
        try:
            self.assertTrue(hx.model.is_torch_extractor('registry_test_extractor'))
            self.assertEqual(
                dict(ext._torch_extractors.metadata('registry_test_extractor')),
                {'license': 'MIT'})
            # Re-registration replaces silently, as before the migration.
            ext.register_torch('registry_test_extractor')(lambda **kw: 'v2')
            self.assertEqual(ext._torch_extractors['registry_test_extractor'](), 'v2')
        finally:
            ext._torch_extractors.unregister('registry_test_extractor')

    def test_mil_registry(self):
        from histox import mil
        from histox.mil import _registry as reg
        self.assertIn('attention_mil', reg.list_models())
        self.assertTrue(reg.is_model('transmil'))
        self.assertIs(reg.get_model_config_class('attention_mil'),
                      reg._mil_models.metadata('attention_mil')['config'])
        config = reg.build_model_config('attention_mil')
        self.assertIsInstance(config, mil.MILModelConfig)
        self.assertEqual(reg._mil_models.extras_package('clam_sb'), 'slideflow-gpl')
        if not reg.is_model('clam_sb'):   # unless the extras package is installed
            with self.assertRaisesRegex(ValueError, 'slideflow-gpl'):
                reg.get_model('clam_sb')
        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            reg.get_model('no_such_model')
        self.assertIn('fastai', reg.list_trainers())

    def test_workflow_registry(self):
        from histox.workflows import (DuplicateRegistrationError, Task,
                                      UnknownTaskError, build_task, list_tasks,
                                      register_task)
        from histox.workflows import _registry as wreg
        self.assertIn('survival', list_tasks())

        @register_task('Registry_Test_Task')
        class RegistryTestTask(Task):
            pass
        try:
            self.assertIn('registry_test_task', list_tasks())
            self.assertIsInstance(build_task('REGISTRY_TEST_TASK'), RegistryTestTask)
            with self.assertRaises(DuplicateRegistrationError):
                register_task('registry_test_task')(RegistryTestTask)
            register_task('registry_test_task', replace=True)(RegistryTestTask)
            with self.assertRaises(UnknownTaskError):
                build_task('no_such_task')
        finally:
            wreg._TASK_REGISTRY.unregister('registry_test_task')


if __name__ == '__main__':
    unittest.main()

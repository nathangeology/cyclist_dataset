from data_science_layer.pipeline.basic_pipeline import BasicPipeline
from data_science_layer.feature_selection.extra_trees_classifier_feature_selection import ExtraTreesClassifierFeatureSelector
from data_science_layer.preprocessing.default_scaler import DefaultScaler
from data_science_layer.preprocessing.default_normalizer import DefaultNormalizer
from data_science_layer.machine_learning.multi_layer_perceptrion_classifier import MultiLayerPerceptronClassifierModel
from data_science_layer.machine_learning.random_forest_classifier import RandomForestClassifierModel
from data_science_layer.machine_learning.gradient_boost_classifier import GradientBoostClassifierModel
from data_science_layer.machine_learning.k_neighbors_classifier import KNeighborsClassifierModel
from data_science_layer.machine_learning.logistic_regression_classifier import LogisticRegressionClassifierModel
from data_science_layer.machine_learning.support_vector_classifier import SupportVectorClassifierModel
from data_science_layer.machine_learning.dummy_classifier import DummyClassifierModel
from data_science_layer.scoring.accuracy_scorer import AccuracyScorer
from data_science_layer.reporting.models_and_metrics_table import PipelineModelMetricReports
from data_science_layer.reporting.extra_trees_classifier_feature_weights import ExtraTreesClassifierFeatureWeightsReport
from data_science_layer.reporting.pair_plot import PairPlot
from data_science_layer.reporting.classifier_report import ClassifierReport
from data_science_layer.reporting.classifier_curves import ClassifierCurves
from data_science_layer.reporting.feature_importance_report import FeatureImportanceReport

class BasicClassifierPipeline(BasicPipeline):
    """Core implementation of a data science pipeline, should
    be extended via subclassing or runtime modification"""

    def __init__(self):
        super().__init__()

        # us = UpSampler()
        # us.random_state = self.random_seed
        # self.add_sampler(sampler=us)
        # ds = DownSampler()
        # ds.random_state = self.random_seed
        # self.add_sampler(sampler=ds)

        # Scorer
        self.scorer = AccuracyScorer()
        # self.set_parametric_scorer(scorer=self.scorer)
        self.dataset_tag = 'basic_cls_pipeline'
        # self.search_type = 'grid'

    def define_default_models(self):
        list_of_models = [
            SupportVectorClassifierModel,
            LogisticRegressionClassifierModel,
            KNeighborsClassifierModel,
            GradientBoostClassifierModel,
            RandomForestClassifierModel,
            MultiLayerPerceptronClassifierModel,
            DummyClassifierModel
        ]
        for model in list_of_models:
            ml = model()
            self.add_ml_model(ml=ml)

    def define_default_preprocessing(self):
        list_of_pre_processors = [
            DefaultScaler,
            DefaultNormalizer
        ]
        for processor in list_of_pre_processors:
            proc = processor()
            self.add_preprocessor(processor=proc)

    def define_default_feature_selector(self):
        etfs = ExtraTreesClassifierFeatureSelector()
        etfs.random_state = self.random_seed
        self.add_feature_selector(feature_selector=etfs)

    def define_default_reports(self):
        list_of_reports = [
            PipelineModelMetricReports,
            ExtraTreesClassifierFeatureWeightsReport,
            PairPlot,
            ClassifierReport,
            ClassifierCurves,
            FeatureImportanceReport
        ]
        for report in list_of_reports:
            rep = report()
            self.add_report(report=rep)

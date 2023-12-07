import copy
import json
import logging
import os
from typing import Dict


logger = logging.getLogger(__name__)
CONFIG_NAME = "config.json"


class PretrainedConfig(object):
    r""" Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.

        Note:
            A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
            It only affects the model's configuration.

        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` with `shortcut names` (string) as keys and `url` (string) of associated pretrained model configurations as values.
            - ``model_type``: a string that identifies the model type, that we serialize into the JSON file, and that we use to recreate the correct object in :class:`~transformers.AutoConfig`.

        Args:
            finetuning_task (:obj:`string` or :obj:`None`, `optional`, defaults to :obj:`None`):
                Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
            num_labels (:obj:`int`, `optional`, defaults to `2`):
                Number of classes to use when the model is a classification model (sequences/tokens)
            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Should the model returns attentions weights.
            output_hidden_states (:obj:`string`, `optional`, defaults to :obj:`False`):
                Should the model returns all hidden-states.
            torchscript (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Is the model used with Torchscript (for PyTorch models).
    """
    pretrained_config_archive_map = {}  # type: Dict[str, str]
    model_type = ""  # type: str

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_past = kwargs.pop("output_past", True)  # Not used by all models
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_decoder = kwargs.pop("is_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.bos_token_id = kwargs.pop("bos_token_id", 0)
        self.pad_token_id = kwargs.pop("pad_token_id", 0)
        self.eos_token_ids = kwargs.pop("eos_token_ids", 0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.id2label = kwargs.pop("id2label", {i: "LABEL_{}".format(i) for i in range(self.num_labels)})
        self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        self.label2id = kwargs.pop("label2id", dict(zip(self.id2label.values(), self.id2label.keys())))
        self.label2id = dict((key, int(value)) for key, value in self.label2id.items())

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def save_pretrained(self, save_directory):
        """
        Save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`string`):
                Directory where the configuration JSON file will be saved.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        r"""

        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration.

        Args:
            pretrained_model_name_or_path (:obj:`string`):
                either:
                  - a string with the `shortcut name` of a pre-trained model configuration to load from cache or
                    download, e.g.: ``bert-base-uncased``.
                  - a string with the `identifier name` of a pre-trained model configuration that was user-uploaded to
                    our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                  - a path to a `directory` containing a configuration file saved using the
                    :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                  - a path or url to a saved configuration JSON `file`, e.g.:
                    ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`string`, `optional`):
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            kwargs (:obj:`Dict[str, any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is
                controlled by the `return_unused_kwargs` keyword parameter.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Force to (re-)download the model weights and configuration files and override the cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.
            proxies (:obj:`Dict`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.:
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`
                The proxies are used on each request.
            return_unused_kwargs: (`optional`) bool:
                If False, then this function returns just the final configuration object.
                If True, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs` is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part
                of kwargs which has not been used to update `config` and is otherwise ignored.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "PretrainedConfig":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "PretrainedConfig":
        """
        Constructs a `Config` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """
        Save this instance to a json file.

        Args:
            json_file_path (:obj:`string`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class BertConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a :class:`~transformers.BertModel`.
        It is used to instantiate an BERT model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 30522):
                Vocabulary size of the BERT model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 3072):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.

        Example::

            from transformers import BertModel, BertConfig

            # Initializing a BERT bert-base-uncased style configuration
            configuration = BertConfig()

            # Initializing a model from the bert-base-uncased style configuration
            model = BertModel(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        split=10,
        num_rnn_layer=1,
        rnn_dropout=0.0,
        rnn_hidden=768,
        rnn="lstm",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.split = split
        self.num_rnn_layer = num_rnn_layer
        self.rnn = rnn
        self.rnn_dropout = rnn_dropout
        self.rnn_hidden = rnn_hidden
import numpy
from dotmap import DotMap

from scenic3d.core.distributions import Distribution, Options
from scenic3d.core.utils import InvalidScenarioError


class ExternalSampler:
    """Abstract class for objects called to sample values for each external parameter.

    Attributes:
        rejectionFeedback: Value passed to the `sample` method when the last sample was rejected.
          This value can be chosen by a Scenic scenario using the global parameter
          ``externalSamplerRejectionFeedback``.
    """

    def __init__(self, params, globalParams):
        # feedback value passed to external sampler when the last scene was rejected
        self.rejectionFeedback = globalParams.get('externalSamplerRejectionFeedback')

    @staticmethod
    def forParameters(params, globalParams):
        """Create an `ExternalSampler` given the sets of external and global parameters.

        The scenario may explicitly select an external sampler by assigning the global
        parameter ``externalSampler`` to a subclass of `ExternalSampler`. Otherwise, a
        `VerifaiSampler` is used by default.

        Args:
            params (tuple): Tuple listing each `ExternalParameter`.
            globalParams (dict): Dictionary of global parameters for the `Scenario`.
              Note that the values of these parameters may be instances of `Distribution`!

        Returns:
            An `ExternalSampler` configured for the given parameters.
        """
        if len(params) > 0:
            externalSampler = globalParams.get('externalSampler', VerifaiSampler)
            if not issubclass(externalSampler, ExternalSampler):
                raise InvalidScenarioError(f'externalSampler type {externalSampler}'
                                           ' not subclass of ExternalSampler')
            return externalSampler(params, globalParams)
        else:
            return None

    def sample(self, feedback):
        """Sample values for all the external parameters.

        Args:
            feedback: Feedback from the last sample (for active samplers).
        """
        self.cachedSample = self.nextSample(feedback)

    def nextSample(self, feedback):
        """Actually do the sampling. Implemented by subclasses."""
        raise NotImplementedError

    def valueFor(self, param):
        """Return the sampled value for a parameter. Implemented by subclasses."""
        raise NotImplementedError


class VerifaiSampler(ExternalSampler):
    """An external sampler exposing the samplers in the VerifAI toolkit.

    The sampler can be configured using the following Scenic global parameters:

        * ``verifaiSamplerType`` -- sampler type (see the ``verifai.server.choose_sampler``
          function); the default is ``'halton'``
        * ``verifaiSamplerParams`` -- ``DotMap`` of options passed to the sampler

    The `VerifaiSampler` supports external parameters which are instances of `VerifaiParameter`.
    """

    def __init__(self, params, globalParams):
        super().__init__(params, globalParams)
        import verifai.server

        # construct FeatureSpace
        usingProbs = False
        self.params = tuple(params)
        for index, param in enumerate(self.params):
            if not isinstance(param, VerifaiParameter):
                raise RuntimeError(f'VerifaiSampler given parameter of wrong type: {param}')
            param.sampler = self
            param.index = index
            if param.probs is not None:
                usingProbs = True
        space = verifai.features.FeatureSpace({
            f'param{index}': verifai.features.Feature(param.domain)
            for index, param in enumerate(self.params)
        })

        # set up VerifAI sampler
        samplerType = globalParams.get('verifaiSamplerType', 'halton')
        samplerParams = globalParams.get('verifaiSamplerParams', None)
        if usingProbs and samplerType == 'ce':
            if samplerParams is None:
                samplerParams = DotMap()
            if 'cont' in samplerParams or 'disc' in samplerParams:
                raise RuntimeError('CE distributions specified in both VerifaiParameters'
                                   'and verifaiSamplerParams')
            cont_buckets = []
            cont_dists = []
            disc_dists = []
            for param in self.params:
                if isinstance(param, VerifaiRange):
                    if param.probs is None:
                        buckets = 5
                        dist = numpy.ones(buckets) / buckets
                    else:
                        dist = numpy.array(param.probs)
                        buckets = len(dist)
                    cont_buckets.append(buckets)
                    cont_dists.append(dist)
                elif isinstance(param, VerifaiDiscreteRange):
                    n = param.high - param.low + 1
                    dist = numpy.ones(n) / n if param.probs is None else numpy.array(param.probs)
                    disc_dists.append(dist)
                else:
                    raise RuntimeError(f'Parameter {param} not supported by CE sampler')
            samplerParams.cont.buckets = cont_buckets
            samplerParams.cont.dist = numpy.array(cont_dists)
            samplerParams.disc.dist = numpy.array(disc_dists)
        _, sampler = verifai.server.choose_sampler(space, samplerType,
                                                   sampler_params=samplerParams)
        self.sampler = sampler

        # default rejection feedback is positive so cross-entropy sampler won't update;
        # for other active samplers an appropriate value should be set manually
        if self.rejectionFeedback is None:
            self.rejectionFeedback = 1

    def nextSample(self, feedback):
        return self.sampler.nextSample(feedback)

    def valueFor(self, param):
        return self.cachedSample[param.index]


class ExternalParameter(Distribution):
    """A value determined by external code rather than Scenic's internal sampler."""

    def __init__(self):
        super().__init__()
        self.sampler = None
        import scenic3d.syntax.veneer as veneer  # TODO improve?
        veneer.registerExternalParameter(self)

    def sampleGiven(self, value):
        """Specialization of  `Samplable.sampleGiven` for external parameters.

        By default, this method simply looks up the value previously sampled by
        `ExternalSampler.sample`.
        """
        assert self.sampler is not None
        return self.sampler.valueFor(self)


class VerifaiParameter(ExternalParameter):
    """An external parameter sampled using one of VerifAI's samplers."""

    def __init__(self, domain):
        super().__init__()
        self.domain = domain

    @staticmethod
    def withPrior(dist, buckets=None):
        """Creates a `VerifaiParameter` using the given distribution as a prior.

        Since the VerifAI cross-entropy sampler currently only supports piecewise-constant
        distributions, if the prior is not of that form it may be approximated. For most
        built-in distributions, the approximation is exact: for a particular distribution,
        check its `bucket` method.
        """
        if not dist.isPrimitive:
            raise RuntimeError('VerifaiParameter.withPrior called on '
                               f'non-primitive distribution {dist}')
        bucketed = dist.bucket(buckets=buckets)
        return VerifaiOptions(bucketed.optWeights if bucketed.optWeights else bucketed.options)


class VerifaiRange(VerifaiParameter):
    """A :obj:`~scenic3d.core.distributions.Range` (real interval) sampled by VerifAI."""

    def __init__(self, low, high, buckets=None, weights=None):
        import verifai.features
        super().__init__(verifai.features.Box([low, high]))
        if weights is not None:
            weights = tuple(weights)
            if buckets is not None and len(weights) != buckets:
                raise RuntimeError(f'VerifaiRange created with {len(weights)} weights '
                                   f'but {buckets} buckets')
        elif buckets is not None:
            weights = [1] * buckets
        else:
            self.probs = None
            return
        total = sum(weights)
        self.probs = tuple(wt / total for wt in weights)

    def sampleGiven(self, value):
        value = super().sampleGiven(value)
        assert len(value) == 1
        return value[0]


class VerifaiDiscreteRange(VerifaiParameter):
    """A :obj:`~scenic3d.core.distributions.DiscreteRange` (integer interval) sampled by VerifAI."""

    def __init__(self, low, high, weights=None):
        import verifai.features
        super().__init__(verifai.features.DiscreteBox([low, high]))
        if weights is not None:
            if len(weights) != (high - low + 1):
                raise RuntimeError(f'VerifaiDiscreteRange created with {len(weights)} weights '
                                   f'for {high - low + 1} values')
            total = sum(weights)
            self.probs = tuple(wt / total for wt in weights)
        else:
            self.probs = None

    def sampleGiven(self, value):
        value = super().sampleGiven(value)
        assert len(value) == 1
        return value[0]


class VerifaiOptions(Options):
    """An :obj:`~scenic3d.core.distributions.Options` (discrete set) sampled by VerifAI."""

    @staticmethod
    def makeSelector(n, weights):
        return VerifaiDiscreteRange(0, n, weights)

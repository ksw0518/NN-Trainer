#include "Tensor.h"
#include <vector>
#include <memory>
struct Layer {
	virtual Tensor forward(const Tensor& input) = 0;
	virtual ~Layer() = default;
};
struct Linear : Layer
{
	Tensor weight;
	Tensor bias;

	Linear(size_t inDimension, size_t outDimension)
		: weight(inDimension, outDimension), bias(outDimension) {}

	Tensor forward(const Tensor& input) override
	{
		size_t inDim = weight.dim(0);
		size_t outDim = weight.dim(1);

		Tensor out(outDim);

		for (size_t o = 0; o < outDim; o++)
		{
			float sum = bias(o);
			for (size_t i = 0; i < inDim; i++)
			{
				sum += input(i) * weight(i, o);
			}
			out(o) = sum;
		}
		return out;
	}
};

struct ReLU : Layer {
	Tensor forward(const Tensor& input) override {
		Tensor out = input;
		for (float& v : out.data)
		{
			v = std::max(0.0f, v);
		}
		return out;
	}
};
struct Network
{
	std::vector<std::unique_ptr<Layer>> layers;
	void addLayer(Layer* layer) {
		layers.emplace_back(layer);
	}
	Tensor forward(const Tensor& input) {
		Tensor x = input;
		for (auto& layer : layers)
			x = layer->forward(x);
		return x;
	}
};
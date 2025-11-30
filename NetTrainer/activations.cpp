#include "activations.h"
#include <algorithm>
namespace activation {

	float relu(float x)
	{
		return std::max(0.0f, x);
	};
	float tanh(float x)
	{
		return tanh(x);
	};

}
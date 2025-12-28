#pragma once
#include <vector>
#include <cassert>
struct Tensor
{
	size_t dimensionality;
	std::vector<size_t> dimensions;
	std::vector<float> data;
	std::vector<size_t> strides;

	Tensor() = default;

	template<typename... Args>
	explicit Tensor(const Args... args) {
		dimensionality = sizeof...(Args);
		dimensions = { static_cast<size_t>(args)... }; // cast to usize
		data.resize((static_cast<size_t>(args) * ...));
		calculateStrides();
	}
	//create 1d tensor
	Tensor(const std::vector<float>& input) {
		dimensionality = 1;
		dimensions.resize(1);
		dimensions[0] = input.size();
		data = input;
		calculateStrides();
	}

	//resize tensor
	template<typename... Args>
	void resize(Args... args) {
		dimensionality = sizeof...(Args);
		dimensions = { args... };
		data.resize((args * ...));
		calculateStrides();
	}
	void resize(const std::vector<uint64_t>& newDims) {
		dimensionality = newDims.size();
		dimensions = newDims;

		uint64_t size = 1;
		for (const size_t d : dimensions)
			size *= d;
		data.resize(size);
		calculateStrides();
	}
	void setDimension(const size_t dimIdx, const size_t newSize) {
		assert(dimIdx < dimensionality);
		dimensions[dimIdx] = newSize;

		uint64_t size = 1;
		for (const size_t d : dimensions)
			size *= d;
		data.resize(size);
		calculateStrides();
	}

	// Add a leading 1 to the dimensions
	void unsqueeze() {
		std::vector<size_t> newSizes(1 + dimensions.size());
		newSizes[0] = 1;
		std::memcpy(newSizes.data() + 1, dimensions.data(), dimensions.size() * sizeof(size_t));

		resize(newSizes);
	}

	float* ptr() { return data.data(); }
	const float* ptr() const { return data.data(); }

	size_t size() const { return data.size(); }
	auto begin() { return data.begin(); }
	auto begin() const { return data.begin(); }
	auto end() { return data.end(); }
	auto end() const { return data.end(); }

	void fill(const float value) {
		std::fill(data.begin(), data.end(), value);
	}

	void calculateStrides() {
		strides.resize(dimensionality);
		if (dimensionality == 0) return;

		strides[dimensionality - 1] = 1;

		for (int i = dimensionality - 2; i >= 0; i--)
			strides[i] = strides[i + 1] * dimensions[i + 1];
	}
	// Get the dimensionality
	auto& dims() { return dimensions; }
	const auto& dims() const { return dimensions; }
	size_t dim(const size_t idx) const { return dimensions[idx]; }

	// Leave the data but change the dimensions
	// assumes the size doesn't change
	void reshape(const std::vector<size_t>& newDims) {
		dimensionality = newDims.size();
		dimensions = newDims;

#ifndef NDEBUG
		uint64_t size = 1;
		for (const size_t d : dimensions)
			size *= d;
		assert(data.size() == size);
#endif

		calculateStrides();
	}

	float& operator()(const size_t i) {
		assert(dimensionality == 1);
		return data[i];
	}

	const float& operator()(const size_t i) const {
		assert(dimensionality == 1);
		return data[i];
	}

	float& operator()(const size_t i, const size_t j) {
		assert(dimensionality == 2);
		return data[i * strides[0] + j];
	}

	const float& operator()(const size_t i, const size_t j) const {
		assert(dimensionality == 2);
		return data[i * strides[0] + j];
	}

	template<typename... Args>
	float& operator()(Args... args) {
		assert(sizeof...(Args) == dimensionality);
		usize idx = 0;
		usize strideIdx = 0;
		((idx += static_cast<usize>(args) * strides[strideIdx++]), ...);
		return data[idx];
	}

	template<typename... Args>
	const float& operator()(Args... args) const {
		assert(sizeof...(Args) == dimensionality);
		usize idx = 0;
		usize strideIdx = 0;
		((idx += static_cast<usize>(args) * strides[strideIdx++]), ...);
		return data[idx];
	}
	Tensor row(size_t r) const {
		Tensor out(dim(1));
		for (size_t c = 0; c < dim(1); c++)
			out(c) = (*this)(r, c);
		return out;
	}
	void setRow(size_t r, const Tensor& row) {
		assert(dimensionality == 2);
		assert(row.dim(0) == dim(1)); // row must match number of columns
		for (size_t c = 0; c < dim(1); c++)
			(*this)(r, c) = row(c);
	}
};
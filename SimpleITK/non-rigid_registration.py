__author__ = "Saumik"
__date__ = "11/03/2023"

import SimpleITK as sitk

# Load the fixed and moving images
fixed_image = sitk.ReadImage('fixed_image.nii', sitk.sitkFloat32)
moving_image = sitk.ReadImage('moving_image.nii', sitk.sitkFloat32)

# Initialize the Image Registration Method
registration_method = sitk.ImageRegistrationMethod()

# Set the Metric as Mutual Information
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

# Set the Optimizer
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Use a B-spline transform (non-rigid)
transformDomainMeshSize=[10]*moving_image.GetDimension()
initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image, 
                                                     transformDomainMeshSize=transformDomainMeshSize, order=3)
registration_method.SetInitialTransform(initial_transform, True)

# Use Linear Interpolation
registration_method.SetInterpolator(sitk.sitkLinear)

# Add Regularization (optional)
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

# Execute the registration
final_transform = registration_method.Execute(fixed_image, moving_image)

# Resample the moving image to the fixed image space
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(final_transform)

resampled_image = resampler.Execute(moving_image)

# Save the result
sitk.WriteImage(resampled_image, 'registered_moving_image.nii')

# Also, save the final transform if needed
sitk.WriteTransform(final_transform, 'final_transform.tfm')

# Function to output iteration information
def command_iteration(method) :
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")

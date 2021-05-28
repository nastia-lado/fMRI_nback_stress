#fMRI decoding

from nilearn import datasets
# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
# 'func' is a list of filenames: one for each subject
fmri_filename = haxby_dataset.func[0]

# print basic information on the dataset
print('First subject functional nifti images (4D) are at: %s' %fmri_filename)  # 4D data

mask_filename = haxby_dataset.mask_vt[0]

# using the subject's anatomical image as a background
from nilearn import plotting
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0], cmap='Paired')

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)

# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_filename)

print(fmri_masked)


print(fmri_masked.shape)

import pandas as pd
# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
print(behavioral)

#Retrieve the experimental conditions, that we are going to use as prediction targets in the decoding
conditions = behavioral['labels']
print(conditions)

#restrict to cats and faces
condition_mask = conditions.isin(['face', 'cat'])

# We apply this mask in the sampe direction to restrict the
# classification to the face vs cat discrimination
fmri_masked = fmri_masked[condition_mask]

print(fmri_masked.shape)

#decoding with svm
from sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)

svc.fit(fmri_masked, conditions)

prediction = svc.predict(fmri_masked)
print(prediction)

#Measuring prediction scores using cross-validation
svc.fit(fmri_masked[:-30], conditions[:-30])

prediction = svc.predict(fmri_masked[-30:])
print((prediction == conditions[-30:]).sum() / float(len(conditions[-30:])))

from sklearn.cross_validation import KFold

cv = KFold(n=len(fmri_masked), n_folds=5)

for train, test in cv:
    conditions_masked = conditions.values[train]
    svc.fit(fmri_masked[train], conditions_masked)
    prediction = svc.predict(fmri_masked[test])
    print((prediction == conditions.values[test]).sum()
           / float(len(conditions.values[test])))
    
from sklearn.cross_validation import cross_val_score
cv_score = cross_val_score(svc, fmri_masked, conditions)
print(cv_score)

cv_score = cross_val_score(svc, fmri_masked, conditions, cv=cv)
print(cv_score)

session_label = behavioral['chunks'][condition_mask]

from sklearn.cross_validation import LeaveOneLabelOut
cv = LeaveOneLabelOut(session_label)
cv_score = cross_val_score(svc, fmri_masked, conditions, cv=cv)
print(cv_score)

#inspecting model wieghts
coef_ = svc.coef_
print(coef_)

print(coef_.shape)

coef_img = masker.inverse_transform(coef_)
print(coef_img)

coef_img.to_filename('haxby_svc_weights.nii.gz')


from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0], title="SVM weights", display_mode="yx")

show()



    
    
    
    
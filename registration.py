# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:10:46 2022

@author: Ilka
@referance: Bhklab. (n.d.). Bhklab/med-imagetools: Transparent and reproducible medical image processing pipelines in python. GitHub. Retrieved July 12, 2022, from https://github.com/bhklab/med-imagetools 
"""

import os, pathlib
import warnings
import datetime
from typing import Optional, Dict, TypeVar
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import pydicom
from pydicom import dcmread
from tifffile import imwrite
from stl import mesh

def load_stl(path):
    x = mesh.Mesh.from_file(path + '.stl')
    x = x.data
    xx = []
    for i in range(len(x)):
        if i < 10:
            print(x[i][1][0])
        xx.append(x[i][1][1])
    x = np.asarray(xx,dtype='int')
    print(np.max(x),np.min(x))
    print(x.shape)
    i,j,k = np.max(x[:,0])-np.min(x[:,0]),np.max(x[:,1])-np.min(x[:,1]),np.max(x[:,2])-np.min(x[:,2])
    mins = np.asarray([np.min(x[:,0]),np.min(x[:,1]),np.min(x[:,2])])
    xx = np.zeros((i+1,j+1,k+1))
    print(xx.shape)
    for i in range(len(x)):
        xx[x[i,0]-mins[0],x[i,1]-mins[1],x[i,2]-mins[2]] = 1.
    x = xx
    print(x.shape)
    for i in range(0,x.shape[2], 5):
        plt.imshow(x[:,:,i].T)
        plt.show()
    return x

T = TypeVar('T')

def read_image(path:str,series_id: Optional[str]=None):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path,seriesID=series_id if series_id else "")
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    return reader.Execute()


def calc_factor(df, type: str):
    '''
    Following the calculation formula stated in https://gist.github.com/pangyuteng/c6a075ba9aa00bb750468c30f13fc603
    '''
    #Fetching some required Meta Data
    try:
        weight = float(df.PatientWeight) * 1000
    except:
        warnings.warn("Patient Weight Not Present. Taking 75Kg")
        weight = 75000
    try:
        scan_time = datetime.datetime.strptime(df.AcquisitionTime, '%H%M%S.%f')
        injection_time = datetime.datetime.strptime(df.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime, '%H%M%S.%f')
        half_life = float(df.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        injected_dose = float(df.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

        # Calculate activity concenteration factor
        a = np.exp(-np.log(2) * ((scan_time - injection_time).seconds / half_life))

        # Calculate SUV factor
        injected_dose_decay = a * injected_dose
    except:
        warnings.warn("Not enough data available, taking average values")
        a = np.exp(-np.log(2) * (1.75 * 3600) / 6588) # 90 min waiting time, 15 min preparation
        injected_dose_decay = 420000000 * a # 420 MBq

    suv = weight/injected_dose_decay
    if type == "SUV":
        return suv
    else:
        return 1/a
    

def from_dicom_pet(path,series_id=None,type="SUV"):
    '''
    Reads the PET scan and returns the data frame and the image dosage in SITK format
    There are two types of existing formats which has to be mentioned in the type
    type:
        SUV: gets the image with each pixel value having SUV value
        ACT: gets the image with each pixel value having activity concentration
    SUV = Activity concenteration/(Injected dose quantity/Body weight)

    Please refer to the pseudocode: https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV) 
    If there is no data on SUV/ACT then backup calculation is done based on the formula in the documentation, although, it may
    have some error.
    '''
    pet      = read_image(path,series_id)
    path_one = pathlib.Path(path,os.listdir(path)[0]).as_posix()
    df       = dcmread(path_one)
    calc     = False
    try:
        if type=="SUV":
            factor = df.to_json_dict()['70531000']["Value"][0]
        else:
            factor = df.to_json_dict()['70531009']['Value'][0]
    except:
        warnings.warn("Scale factor not available in DICOMs. Calculating based on metadata, may contain errors")
        factor = calc_factor(df,type)
        calc = True
    img_pet = sitk.Cast(pet, sitk.sitkFloat32)

    #SimpleITK reads some pixel values as negative but with correct value
    img_pet = sitk.Abs(img_pet * factor)

    metadata = {}
    return img_pet, df, factor, calc, metadata

class PET(sitk.Image):
    def __init__(self, img_pet, df, factor, calc, metadata: Optional[Dict[str, T]] = None):
        super().__init__(img_pet)
        self.img_pet = img_pet
        self.df = df
        self.factor = factor
        self.calc = calc
        self.resampled_pt = None
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = {}
    
    @classmethod
    def from_dicom_pet(cls, path,series_id=None,type="SUV"):
        '''
        Reads the PET scan and returns the data frame and the image dosage in SITK format
        There are two types of existing formats which has to be mentioned in the type
        type:
            SUV: gets the image with each pixel value having SUV value
            ACT: gets the image with each pixel value having activity concentration
        SUV = Activity concenteration/(Injected dose quantity/Body weight)

        Please refer to the pseudocode: https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV) 
        If there is no data on SUV/ACT then backup calculation is done based on the formula in the documentation, although, it may
        have some error.
        '''
        pet      = read_image(path,series_id)
        path_one = pathlib.Path(path,os.listdir(path)[0]).as_posix()
        df       = dcmread(path_one)
        calc     = False
        try:
            if type=="SUV":
                factor = df.to_json_dict()['70531000']["Value"][0]
            else:
                factor = df.to_json_dict()['70531009']['Value'][0]
        except:
            warnings.warn("Scale factor not available in DICOMs. Calculating based on metadata, may contain errors")
            factor = cls.calc_factor(df,type)
            calc = True
        img_pet = sitk.Cast(pet, sitk.sitkFloat32)

        #SimpleITK reads some pixel values as negative but with correct value
        img_pet = sitk.Abs(img_pet * factor)

        metadata = {}
        return cls(img_pet, df, factor, calc, metadata)
        # return cls(img_pet, df, factor, calc)
        
    def get_metadata(self):
        '''
        Forms the important metadata for reference in the dictionary format
        {
            scan_time (in seconds): AcquisitionTime 
            injection_time (in seconds): RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            weight (in kg): PatientWeight
            half_life (in seconds): RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
            injected_dose: RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            Values_Assumed: True when some values are not available and are assumed for the calculation of SUV factor
            factor: factor used for rescaling to bring it to SUV or ACT
        }
        '''
        self.metadata = {}
        try:
            self.metadata["weight"] = float(self.df.PatientWeight)
        except:
            pass
        try:
            self.metadata["scan_time"] = datetime.datetime.strptime(self.df.AcquisitionTime, '%H%M%S.%f')
            self.metadata["injection_time"] = datetime.datetime.strptime(self.df.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime, '%H%M%S.%f')
            self.metadata["half_life"] = float(self.df.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            self.metadata["injected_dose"] = float(self.df.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        except:
            pass
        self.metadata["factor"] = self.factor
        self.metadata["Values_Assumed"] = self.calc
        return self.metadata

    def resample_pet(self,
                     ct_scan: sitk.Image) -> sitk.Image:
        '''
        Resamples the PET scan so that it can be overlayed with CT scan. The beginning and end slices of the 
        resampled PET scan might be empty due to the interpolation
        '''
        self.resampled_pt = sitk.Resample(self.img_pet, ct_scan)#, interpolator=sitk.sitkNearestNeighbor) # commented interporator due to error       
        return self.resampled_pt

    def show_overlay(self,
                     ct_scan: sitk.Image,
                     slice_number: int) -> plt.figure:
        '''
        For a given slice number, the function resamples PET scan and overlays on top of the CT scan and returns the figure of the
        overlay
        '''
        resampled_pt = self.resample_pet(ct_scan)
        plt.figure("Overlayed image", figsize=[15, 10])
              
        plt.subplot(1,3,1)
        pt_arr = sitk.GetArrayFromImage(resampled_pt)
        #print(pt_arr.shape)
        plt.imshow(pt_arr[slice_number,:,:])
        plt.title('PT slice'+str(slice_number))
        
        plt.subplot(1,3,2)
        ct_arr = sitk.GetArrayFromImage(ct_scan)
        #print(ct_arr.shape)
        plt.imshow(ct_arr[slice_number,:,:])
        plt.title('CT slice'+str(slice_number))
        
        plt.subplot(1,3,3)
        plt.imshow(ct_arr[slice_number,:,:], cmap=plt.cm.gray)
        plt.imshow(pt_arr[slice_number,:,:], cmap=plt.cm.hot, alpha=.4)
        plt.title('Overlay slice'+str(slice_number))
        
        plt.show()
        

    @staticmethod
    def calc_factor(df, type: str):
        '''
        Following the calculation formula stated in https://gist.github.com/pangyuteng/c6a075ba9aa00bb750468c30f13fc603
        '''
        #Fetching some required Meta Data
        try:
            weight = float(df.PatientWeight) * 1000
        except:
            warnings.warn("Patient Weight Not Present. Taking 75Kg")
            weight = 75000
        try:
            scan_time = datetime.datetime.strptime(df.AcquisitionTime, '%H%M%S.%f')
            injection_time = datetime.datetime.strptime(df.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime, '%H%M%S.%f')
            half_life = float(df.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            injected_dose = float(df.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

            # Calculate activity concenteration factor
            a = np.exp(-np.log(2) * ((scan_time - injection_time).seconds / half_life))

            # Calculate SUV factor
            injected_dose_decay = a * injected_dose
        except:
            warnings.warn("Not enough data available, taking average values")
            a = np.exp(-np.log(2) * (1.75 * 3600) / 6588) # 90 min waiting time, 15 min preparation
            injected_dose_decay = 420000000 * a # 420 MBq

        suv = weight/injected_dose_decay
        if type == "SUV":
            return suv
        else:
            return 1/a
       
        
def load_meta(path):
    meta = str(pydicom.read_file(path + '.dcm'))
    return meta


def save_as_tif(path,x,meta):
    #Image.fromarray(x).save(path + 'combined.tif',description = meta)
    imwrite(path + 'combined.tif', x,description=meta,)
    return None


def norm(x):
    return((x - np.min(x))/(np.max(x) - np.min(x)))
        
        
def combine_as_tif(out_path, path_stl, path_ct, path_pet, dcm_sample_ct = None, dcm_sample_pet = None):
    meta = None
    
    img_ct,_,_,_,_ = from_dicom_pet(path_ct)
    
    img_pet, df, factor, calc, metadata = from_dicom_pet(path_pet)
    
    pet_warper = PET(img_pet, df, factor, calc, metadata)
    
    img_pet = pet_warper.resample_pet(img_ct)
    
    ct = sitk.GetArrayFromImage(img_ct)
    
    pet = sitk.GetArrayFromImage(img_pet)
    
    
    if path_stl != None:
        stl = load_stl(path_stl)
        print(stl.shape)
    else:
        stl = np.zeros(ct.shape)
    
    x = np.asarray([norm(pet)*255,norm(ct)*255,norm(stl)*255])
    x = x.astype('uint16')
    print(x.shape)
    
    if dcm_sample_ct != None:
        meta = load_meta(path_ct + dcm_sample_ct)
        
    if dcm_sample_pet != None:
        meta = meta + load_meta(path_pet + dcm_sample_pet)
        
    save_as_tif(out_path,x,meta)
    
    return x
    
        
if __name__ == "__main__":
    #path_stl = 'C:/Users/Ilka/Desktop/Northwestern/Hope rotation/WholeBody'
    path_stl = None
    path_ct = 'C:/Users/Ilka/Desktop/Northwestern/Hope rotation/ct scan/RM-S302 NIRC-8791-2001_S302-Zr89-VRC01-aCARIAS-24h-5Nov2020_CT_2020-11-05_115133_._LDCT_n200__00000'
    ct_img = '/1.2.840.113704.1.111.2148.1604598918.175'
    path_pet = 'C:/Users/Ilka/Desktop/Northwestern/Hope rotation/pt scan/RM-S302 NIRC-8791-2001_S302-Zr89-VRC01-aCARIAS-24h-5Nov2020_PT_2020-11-05_115710_._(WB.CTAC).Body_n150__00000'
    pet_img = '/1.3.46.670589.28.2.15.4.9181.50123.3.580.0.1604600691'
    path_out = 'C:/Users/Ilka/Desktop/Northwestern/Hope rotation/'
    x = combine_as_tif(path_out,path_stl,path_ct,path_pet,ct_img,pet_img)   
    for i in range(x.shape[1]):
        plt.imshow(x[:,i].T)
        plt.show()

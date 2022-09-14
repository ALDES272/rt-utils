import os
from typing import List
from enum import IntEnum

import cv2 as cv
import numpy as np

from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

# from skimage.measure import label
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from skimage import measure
# import ast
from collections import Counter
from rt_utils.utils import ROIData#, SOPClassUID


def load_sorted_image_series(dicom_series_path: str):
    """
    File contains helper methods for loading / formatting DICOM images and contours
    """

    series_data = load_dcm_images_from_path(dicom_series_path)

    if len(series_data) == 0:
        raise Exception("No DICOM Images found in input path")

    # Sort slices in ascending order
    series_data.sort(key=get_slice_position, reverse=False)

    return series_data


def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    series_data = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, "pixel_array"):
                    series_data.append(ds)

            except Exception:
                # Not a valid DICOM file
                continue

    return series_data


def get_contours_coords(roi_data: ROIData, series_data):
    transformation_matrix = get_pixel_to_patient_transformation_matrix(series_data)
    # print("transformation_matrix",transformation_matrix)
    series_contours = []
    for i, series_slice in enumerate(series_data):
        # print("slice \n",i)
        mask_slice = roi_data.mask[:, :, i]

        # Do not add ROI's for blank slices
        if np.sum(mask_slice) == 0:
            series_contours.append([])
            # print("Skipping empty mask layer")
            continue

        # Create pin hole mask if specified
        if roi_data.use_pin_hole:
            mask_slice = create_pin_hole_mask(mask_slice, roi_data.approximate_contours)

        # Get contours from mask
        contours, _ = find_mask_contours(mask_slice, roi_data.approximate_contours)
        validate_contours(contours)
        # print("nb contours ", len(contours))
        # print("contours\n",contours)
        # Format for DICOM
        formatted_contours = []
        for contour in contours:
            # Add z index
            contour = np.concatenate(
                (np.array(contour), np.full((len(contour), 1), i)), axis=1
            )

            transformed_contour = apply_transformation_to_3d_points(
                contour, transformation_matrix
            )
            dicom_formatted_contour = np.ravel(transformed_contour).tolist()
            # dicom_formatted_contour = [ round(elem, 2) for elem in dicom_formatted_contour ]
            formatted_contours.append(dicom_formatted_contour)
            # print("formatted_contours",formatted_contours)
        series_contours.append(formatted_contours)

    return series_contours

def HierarcHy_Level_Det(HHD_hierarchy):
    #returns list of deepness level in the image
    #first-level images are at -1
    outlist=[]
    for iHLd in range(len(HHD_hierarchy)):
        if HHD_hierarchy[iHLd][3]== -1:
            outlist.append(-1)
        else :
            # ref_level=HHD_hierarchy[iHLd][3]
            outlist.append(outlist[HHD_hierarchy[iHLd][3]]-1)
    return(outlist)

def MY_Intermediate_Pixels(IP_coord0,IP_coord1):
    """
    Returns shared corner coordinates of two neighbor pixels :
    [coord_x,coord_y]
    """
    # print("neighbor pixels :",IP_coord0,IP_coord1)
    if IP_coord0[0]==IP_coord1[0]:
        Sc_Coord=float((IP_coord0[1]+IP_coord1[1])/2)
        return([[IP_coord0[0]-0.5,Sc_Coord],[IP_coord0[0]+0.5,Sc_Coord]])
    elif IP_coord0[1]==IP_coord1[1]:
        Sc_Coord=float((IP_coord0[0]+IP_coord1[0])/2)
        return([[Sc_Coord,IP_coord0[1]-0.5],[Sc_Coord,IP_coord0[1]+0.5]])
    
def MY_Intra_Contour_Detector(ICDpair_coord, ICD_mask):
    Exit_Bool=False
    ICD_coo1=ICDpair_coord[0]
    ICD_coo2=ICDpair_coord[1]
    if ICD_coo1[0]== ICD_coo2[0]:
        Neigh_pixel1=[int(ICD_coo1[0]-0.5),int((ICD_coo1[1]+ICD_coo2[1])/2)]
        Neigh_pixel2=[int(ICD_coo1[0]+0.5),int((ICD_coo1[1]+ICD_coo2[1])/2)]
    elif ICD_coo1[1]== ICD_coo2[1]:
        Neigh_pixel1=[int((ICD_coo1[0]+ICD_coo2[0])/2),int(ICD_coo1[1]-0.5)]
        Neigh_pixel2=[int((ICD_coo1[0]+ICD_coo2[0])/2),int(ICD_coo1[1]+0.5)]
    # print("Neigh_pixel1",Neigh_pixel1)
    # print("Neigh_pixel2",Neigh_pixel2)
    if ICD_mask[Neigh_pixel1[1],Neigh_pixel1[0]] == 0 and ICD_mask[Neigh_pixel2[1],Neigh_pixel2[0]] == 0:
        Exit_Bool = True
            
    return Exit_Bool
    
   
   
def MY_Intermediate_Contours_Generator(ICG_coord,ICG_mask):
    """
    For a given coordonate ICG_coord, finds neighbor pixels in ICG_mask that
    are non-negative (thus part of contours).
    Then for each of these pixels, returns the pair of shared corner coordonates.
    (function Intermediate_Pixels)
    Can get 1 to 4 pair of shared corner coordonates. 
    If nb of pair >1 : there are redundant corners that are removed.
    Returns a final list of corner pixels

    Parameters
    ----------
    ICG_coord : [coord,coord]
        a coordonate of the dilated mask
    ICG_mask : 2D numpy array
        DESCRIPTION.

    Returns
    -------
    a list of coordonates

    """
    List_coords=[]
    if ICG_mask[ICG_coord[1],ICG_coord[0]-1] == 0:
        List_coords.append(MY_Intermediate_Pixels([ICG_coord[0],ICG_coord[1]],[ICG_coord[0]-1,ICG_coord[1]]))
    if ICG_mask[ICG_coord[1],ICG_coord[0]+1] == 0:
        List_coords.append(MY_Intermediate_Pixels([ICG_coord[0],ICG_coord[1]],[ICG_coord[0]+1,ICG_coord[1]]))
    if ICG_mask[ICG_coord[1]+1,ICG_coord[0]] == 0:
        List_coords.append(MY_Intermediate_Pixels([ICG_coord[0],ICG_coord[1]],[ICG_coord[0],ICG_coord[1]+1]))
    if ICG_mask[ICG_coord[1]-1,ICG_coord[0]] == 0:
        List_coords.append(MY_Intermediate_Pixels([ICG_coord[0],ICG_coord[1]],[ICG_coord[0],ICG_coord[1]-1]))
    # print("List_coords: ",List_coords)

    List_coords2=[]
    #remove pairs of points that cross the mask
    for ICG_i in range(len(List_coords)):
        if MY_Intra_Contour_Detector(List_coords[ICG_i], ICG_mask)==False:
            List_coords2.append(List_coords[ICG_i])

    return(List_coords2)

def MY_complete_list_coordonates(CLC_contours,CLC_mask):
    #function taking a lot of time
    CLC = [[]]
    for coordonate in range(len(CLC_contours)) :
        Currcoordonate=CLC_contours[coordonate]
        CLC[0].append(Currcoordonate)
        if Currcoordonate != CLC_contours[-1] : #not last element in current list of contour
            Nextcoordonate=CLC_contours[coordonate+1]
        else :
            Nextcoordonate=CLC_contours[0] #takes first element of current coord list
        # print("Currcoordonate",Currcoordonate)
        # print("Nextcoordonate",Nextcoordonate)
        if abs(Currcoordonate[0]-Nextcoordonate[0])+abs(Currcoordonate[1]-Nextcoordonate[1])==2: #diagonal points
            DiagoCoord1=[Currcoordonate[0],Nextcoordonate[1]]
            DiagoCoord2=[Nextcoordonate[0],Currcoordonate[1]]
            if CLC_mask[Nextcoordonate[1],Currcoordonate[0]] != 0 and CLC_mask[Currcoordonate[1],Nextcoordonate[0]] == 0:
               CLC[0].append(DiagoCoord1)
            elif CLC_mask[Nextcoordonate[1],Currcoordonate[0]] == 0 and CLC_mask[Currcoordonate[1],Nextcoordonate[0]] != 0:
               CLC[0].append(DiagoCoord2)
            elif CLC_mask[Nextcoordonate[1],Currcoordonate[0]] == 0 and CLC_mask[Currcoordonate[1],Nextcoordonate[0]] == 0:
               raise Exception("Problem : les deux diagos sont nulles")
            elif CLC_mask[Nextcoordonate[1],Currcoordonate[0]] != 0 and CLC_mask[Currcoordonate[1],Nextcoordonate[0]] != 0:
               raise Exception("Problem : les deux diagos sont non-nulles")   
    return(CLC)


def MY_check_corner(cc_co1,cc_co2,cc_co3,cc_mask):
    """
    Returns
    -------
    cc_Bool : Boolean
       Returns False if the candidate should not be considered
    """
    cc_Bool=True
    if cc_co1[0] == cc_co2[0] and cc_co1[0] == cc_co3[0] :
        cc_Bool=False
    elif cc_co1[1] == cc_co2[1] and cc_co1[1] == cc_co3[1] :
        cc_Bool=False
    else :
        cellA=[(cc_co1[0]+cc_co3[0])/2,(cc_co1[1]+cc_co3[1])/2]
        if cc_co1[0] == cc_co2[0] and cc_co2[1] == cc_co3[1] :
            cellB=[int(2*cc_co1[0]-cellA[0]),int(2*cc_co2[1]-cellA[1])]
        elif cc_co1[1] == cc_co2[1] and cc_co2[0] == cc_co3[0] :
            cellB=[int(2*cc_co2[0]-cellA[0]),int(2*cc_co1[1]-cellA[1])]
        else : raise Exception("PROBLEM : not any possibility")
        if cc_mask[cellB[1],cellB[0]] !=0 :
            cc_Bool=False

    return cc_Bool

def MY_contour_simplificator(cs_contour):
    #simplifies the contour to draw straight lines where necessary
    if len(cs_contour)>2:
        out_cs_contour=[cs_contour[0],cs_contour[1]]
        for cs in range(2,len(cs_contour)):
            new_candi=cs_contour[cs]
            old_cand0=cs_contour[cs-2]
            old_cand1=cs_contour[cs-1]
            if old_cand0[0]==old_cand1[0] and old_cand0[0]==new_candi[0]:
                out_cs_contour[-1]=new_candi
            elif old_cand0[1]==old_cand1[1] and old_cand0[1]==new_candi[1]:
                out_cs_contour[-1]=new_candi
            else:out_cs_contour.append(new_candi)
    return(out_cs_contour)


def MY_contours_initiator(mask,approximate_contours):
    #this function recreates the contours
    # plt.imshow(mask)
    # plt.title("Mask")
    # plt.show()
    approximation_method = (
        cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv.findContours(
        mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
    
    
    inverse_mask=1-mask
    inverse_contours, hinverse_ierarchy = cv.findContours(
        inverse_mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
    
    init_contours=contours
    contours = list(contours
    )  # Open-CV updated contours to be a tuple so we convert it back into a list here
    
    
    init_inverse_contours=inverse_contours
    inverse_contours = list(inverse_contours
    )  # Open-CV updated contours to be a tuple so we convert it back into a list here
    
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    for i, contour in enumerate(inverse_contours):
        inverse_contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
        
    
    
    hierarchy = hierarchy[0]  # Format extra array out of data

    PersonnalizedHier=HierarcHy_Level_Det(hierarchy)
    if approximate_contours ==False :
        contour_initiato_list = [[] for i in range(len(contours))]
        masks_contour_intiato_list=[]
        for contour in range(len(contours)) :

            if PersonnalizedHier[contour]%2 !=0 :
                #not a hole structure
                img_contours = np.zeros(mask.shape)
                cv.drawContours(img_contours, init_contours, contour, (1,0,0), 1)
                unique_contour_mask = ndimage.binary_fill_holes(img_contours).astype(int)
                unique_contours, u_hierarchy = cv.findContours(unique_contour_mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
           
            if PersonnalizedHier[contour]%2 ==0 :
                #hole structure
                img_contours = np.zeros(mask.shape)
                cv.drawContours(img_contours, init_contours, contour, (1,0,0), 1)
                # plt.imshow(img_contours)
                # plt.title("img_contours")
                # plt.show()
                # plt.imshow(1-img_contours)
                # plt.title("1-img_contours")
                # plt.show()
                
                # plt.imshow(ndimage.binary_fill_holes(1-img_contours))
                # plt.title("ndimage.binary_fill_holes(1-img_contours)")
                # plt.show()
                # plt.imshow(ndimage.binary_fill_holes(img_contours).astype(int))
                # plt.title("ndimage.binary_fill_holes(img_contours).astype(int)")
                # plt.show()
                unique_contour_mask = ndimage.binary_fill_holes(img_contours).astype(int)-img_contours
                
                    
                # unique_contour_mask = img_contours
                # unique_contour_mask = ndimage.binary_fill_holes(img_contours).astype(int)#-img_contours
                unique_contour_mask=unique_contour_mask.astype(int)
                unique_contours, u_hierarchy = cv.findContours(unique_contour_mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
            # plt.imshow(unique_contour_mask)
            # plt.title("unique_contour_mask")
            # plt.show()

            unique_contours = list(unique_contours)  
            # Open-CV updated contours to be a tuple so we convert it back into a list here
            
            contour_initiato_list[contour]=unique_contours
            masks_contour_intiato_list.append(unique_contour_mask)
        return(contour_initiato_list,hierarchy,masks_contour_intiato_list)


def new_MY_contours_initiator(mask,approximate_contours):
    #this function recreates the contours
    # plt.imshow(mask)
    # plt.title("Mask")
    # plt.show()
    approximation_method = (
        cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv.findContours(
        mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
    
    
    inverse_mask=1-mask
    
    # plt.imshow(inverse_mask)
    # plt.title("inverse_mask")
    # plt.show()
    
    inverse_contours, hinverse_ierarchy = cv.findContours(
        inverse_mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
    
    init_contours=contours
    contours = list(contours)  # Open-CV updated contours to be a tuple so we convert it back into a list here
    
    
    init_inverse_contours=inverse_contours
    inverse_contours = list(inverse_contours)  
    
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    for i, contour in enumerate(inverse_contours):
        inverse_contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
        
    
    
    hierarchy = hierarchy[0]  # Format extra array out of data
    # print("hierarchy\n",hierarchy)
    
    PersonnalizedHier=HierarcHy_Level_Det(hierarchy)
    # print("PersonnalizedHier\n",PersonnalizedHier)
    
    hinverse_ierarchy = hinverse_ierarchy[0]
    
    # print("hinverse_ierarchy\n",hinverse_ierarchy)
    PersonnalizedInverseHier=HierarcHy_Level_Det(hinverse_ierarchy)
    # print("PersonnalizedInverseHier\n",PersonnalizedInverseHier)
    
    
    if approximate_contours ==False :
        contour_initiato_list = []
        masks_contour_intiato_list=[]
        for contour in range(len(PersonnalizedHier)) :
            if PersonnalizedHier[contour]%2 !=0 :
                #not a hole structure
                img_contours = np.zeros(mask.shape)
                cv.drawContours(img_contours, init_contours, contour, (1,0,0), 1)
                unique_contour_mask = ndimage.binary_fill_holes(img_contours).astype(int)
                unique_contours, u_hierarchy = cv.findContours(unique_contour_mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
                
                unique_contours = list(unique_contours)  
                # Open-CV updated contours to be a tuple so we convert it back into a list here
                
                contour_initiato_list.append(unique_contours)
                masks_contour_intiato_list.append(unique_contour_mask)
        for contour in range(1,len(PersonnalizedInverseHier)) :
            if PersonnalizedInverseHier[contour]%2 !=0 :
                #hole structure
                img_contours = np.zeros(inverse_mask.shape)
                cv.drawContours(img_contours, init_inverse_contours, contour, (1,0,0), 1)
                unique_contour_mask = ndimage.binary_fill_holes(img_contours).astype(int)
                unique_contour_mask=unique_contour_mask.astype(int)
                unique_contours, u_hierarchy = cv.findContours(unique_contour_mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
    
                unique_contours = list(unique_contours)  
                # Open-CV updated contours to be a tuple so we convert it back into a list here
                
                contour_initiato_list.append(unique_contours)
                masks_contour_intiato_list.append(unique_contour_mask)
        return(contour_initiato_list,hierarchy,masks_contour_intiato_list)


def CornerFinder(contour_CF):
    outlist=[contour_CF[0]]
    list_corners=[]
    for coord in range(1,len(contour_CF)):
        previousx=outlist[-1][0]
        previousy=outlist[-1][1]
        currentx=contour_CF[coord][0]
        currenty=contour_CF[coord][1]
        
        if previousx == currentx or previousy==currenty:
            outlist.append(contour_CF[coord])
        else:
            if previousx.is_integer() and currenty.is_integer():
                list_corners.append([currentx,previousy])
                outlist.append([currentx,previousy])
                outlist.append(contour_CF[coord])
            elif currentx.is_integer() and previousy.is_integer():
                list_corners.append([previousx,currenty])
                outlist.append([previousx,currenty])
                outlist.append(contour_CF[coord])
            else : print("!!!Problem no match")
    return(list_corners)

def find_mask_contours(mask: np.ndarray, approximate_contours: bool):
    #new version that uses skimage.measure to find contours instead of shitty opencv
    contours_AD = measure.find_contours(mask)   
    contours_AD=list(contours_AD)
    # hierarchy = list([ [ len(contours_AD) ] * len(contours_AD) ] * len(contours_AD))
    hierarchy = [[ len(contours_AD)  for __ in range(4)] for _ in range(len(contours_AD) )]
    
    out_contours=[]
    for iAD, contour_AD in enumerate(contours_AD):
        contours_AD[iAD] = [[pos[1], pos[0]] for pos in contour_AD]
        
    for iAD in range(len(contours_AD)):
        NewList=CornerFinder(contours_AD[iAD])
        out_contours.append(NewList)
    # print("out_contours",out_contours)
    # print("hierarchy",hierarchy)
    return(out_contours,hierarchy)
    
def old_1_find_mask_contours(mask: np.ndarray, approximate_contours: bool):
    contour_initiato_list,hierarchy,masks_contour_intiato_list=new_MY_contours_initiator(mask,approximate_contours)     
    # print("contour_initiato_list",contour_initiato_list)
    end_contour_list = [[] for i in range(len(contour_initiato_list))]
    for contour in range(len(contour_initiato_list)):
        unique_contours=contour_initiato_list[contour]
        unique_contour_mask=masks_contour_intiato_list[contour]
        # plt.imshow(unique_contour_mask)
        # plt.title("unique_contour_mask")
        # plt.show()
        for i, contour_ in enumerate(unique_contours):
            unique_contours[i] = [[pos[0][0], pos[0][1]] for pos in contour_]

        #assigns corner values :
        Intermediate_pair_list=[]
        for coordonate in range(len(unique_contours[0])) :
            Currcoordonate=unique_contours[0][coordonate]
            New_Corner_coordonates=MY_Intermediate_Contours_Generator(Currcoordonate,unique_contour_mask)
            for fmc_pair in range(len(New_Corner_coordonates)):
                if New_Corner_coordonates[fmc_pair] not in Intermediate_pair_list:
                    Intermediate_pair_list.append(New_Corner_coordonates[fmc_pair])
        End_List=[Intermediate_pair_list[0][0],Intermediate_pair_list[0][1]]
       
        Col0=[elem[0] for elem in Intermediate_pair_list]
        Col1=[elem[1] for elem in Intermediate_pair_list]

        Col0.pop(0)
        Col1.pop(0)
        
        # UniqValues=Counter(tuple(e) for e in Col0+Col1)
        
        # print("Col0 - length :{}\n".format(len(Col0)),Col0)
        # print("Col1 - length :{}\n".format(len(Col1)),Col1)
        # print("End_List\n",End_List)
        while len(Col0) !=0 and len(Col1) !=0 :
            Candi_Col0=len(Intermediate_pair_list)+1
            Candi_Col1=len(Intermediate_pair_list)+1
            
            if (End_List[-1])in Col0 : 
                Candi_Col0=Col0.index(End_List[-1])
            if End_List[-1]in Col1 : 
                Candi_Col1=Col1.index(End_List[-1])
            if End_List[-1] not in Col1 and End_List[-1] not in Col0:
                print("End_List[-1] not in Col1 and End_List[-1] not in Col0")
                print("End_List[-1]",End_List[-1] )
            if Candi_Col0<=Candi_Col1:
                try:
                    Candi_Col_Index=Candi_Col0
                    Candi_Coordonate=Col1[Candi_Col_Index]
                except IndexError:
                   
                    plt.imshow(unique_contour_mask)
                    plt.title("Error image")
                    plt.savefig('Error_image.pdf',dpi=300)
                    plt.show()
                  
                    print("Candi_Col0",Candi_Col0)
                    print("Candi_Col1",Candi_Col1)
                    print("End_List ",End_List )
                    print("Col0\n",Col0)
                    print("Col1\n",Col1)
                    print("Candi_Col_Index\n",Candi_Col_Index)
            else :
                try:
                    Candi_Col_Index=Candi_Col1
                    Candi_Coordonate=Col0[Candi_Col_Index]
                except IndexError:
                    
                    plt.imshow(unique_contour_mask)
                    plt.title("Error image")
                    plt.savefig('Error_image.pdf',dpi=300)
                    plt.show()
                    
                    print("Candi_Col0",Candi_Col0)
                    print("Candi_Col1",Candi_Col1)
                    print("End_List ",End_List )
                    print("Col0\n",Col0)
                    print("Col1\n",Col1)
                    print("Candi_Col_Index\n",Candi_Col_Index)
            UniqValuesRemaining=Counter(tuple(e) for e in Col0+Col1)
            CHECKCORNER =True
            if UniqValuesRemaining[(End_List[-1][0], End_List[-1][1])] >1 :
                 CHECKCORNER=MY_check_corner(End_List[-2],End_List[-1],Candi_Coordonate,unique_contour_mask)
            if CHECKCORNER ==False :
                #reassigns the candidates later in the list
                Col1.append(Col1[Candi_Col_Index])
                Col0.append(Col0[Candi_Col_Index])
                Col1.pop(Candi_Col_Index)
                Col0.pop(Candi_Col_Index)
            else :
                End_List.append(Candi_Coordonate)
                Col0.pop(Candi_Col_Index)
                Col1.pop(Candi_Col_Index)
        end_contour_list[contour]=End_List[:-1]
        # print("end_contour_list[contour] before simplification\n",end_contour_list[contour])
        end_contour_list[contour]=MY_contour_simplificator(end_contour_list[contour])
        print("end_contour_list[contour] after simplification\n",end_contour_list[contour])
        print("end_contour_list",end_contour_list)
    return end_contour_list, hierarchy


def old_find_mask_contours(mask: np.ndarray, approximate_contours: bool):
    # plt.imshow(mask)
    # plt.colorbar()
    # plt.title(np.unique(mask))
    # plt.show()
    approximation_method = (
        cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE
    )
    contours, hierarchy = cv.findContours(
        mask.astype(np.uint8), cv.RETR_TREE, approximation_method
    )
    # Format extra array out of data
    contours = list(
        contours
    )  # Open-CV updated contours to be a tuple so we convert it back into a list here
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    hierarchy = hierarchy[0]  # Format extra array out of data
    
    # Implémentation personnelle pour rendre RTStruct plus précis :
    if approximate_contours ==False :
        # print("entering my code segment")
        end_contour_list = [[] for i in range(len(contours))]
        for contour in range(len(contours)) :
            for coordonate in range(len(contours[contour])) :
                Currcoordonate=contours[contour][coordonate]
                end_contour_list[contour].append(Currcoordonate)
                if Currcoordonate != contours[contour][-1] : #not last element in current list of contour
                    Nextcoordonate=contours[contour][coordonate+1]
                else :
                    Nextcoordonate=contours[contour][0] #takes first element of current coord list
                if abs(Currcoordonate[0]-Nextcoordonate[0])+abs(Currcoordonate[1]-Nextcoordonate[1])==2: #diagonal points
                    DiagoCoord1=[Currcoordonate[0],Nextcoordonate[1]]
                    DiagoCoord2=[Nextcoordonate[0],Currcoordonate[1]]
                    if mask[Nextcoordonate[1],Currcoordonate[0]] != 0 and mask[Currcoordonate[1],Nextcoordonate[0]] == 0:
                       end_contour_list[contour].append(DiagoCoord1)
                    elif mask[Nextcoordonate[1],Currcoordonate[0]] == 0 and mask[Currcoordonate[1],Nextcoordonate[0]] != 0:
                       end_contour_list[contour].append(DiagoCoord2)
                    elif mask[Nextcoordonate[1],Currcoordonate[0]] == 0 and mask[Currcoordonate[1],Nextcoordonate[0]] == 0:
                       raise Exception("Problem : les deux diagos sont nulles")
                    elif mask[Nextcoordonate[1],Currcoordonate[0]] != 0 and mask[Currcoordonate[1],Nextcoordonate[0]] != 0:
                       raise Exception("Problem : les deux diagos sont non-nulles")   
    else : end_contour_list=contours
    # print(end_contour_list)
    return end_contour_list, hierarchy


def create_pin_hole_mask(mask: np.ndarray, approximate_contours: bool):
    """
    Creates masks with pin holes added to contour regions with holes.
    This is done so that a given region can be represented by a single contour.
    """

    contours, hierarchy = find_mask_contours(mask, approximate_contours)
    print("contours\n",contours)
    print("hierarchy\n",hierarchy)
    pin_hole_mask = mask.copy()

    # Iterate through the hierarchy, for child nodes, draw a line upwards from the first point
    for i, array in enumerate(hierarchy):
        parent_contour_index = array[Hierarchy.parent_node]
        print("parent_contour_index\n",parent_contour_index)
        if parent_contour_index == -1:
            continue  # Contour is not a child

        child_contour = contours[i]
        print("child_contour\n",child_contour)

        line_start = tuple(child_contour[0])
        print("line_start\n",line_start)

        pin_hole_mask = draw_line_upwards_from_point(
            pin_hole_mask, line_start, fill_value=0
        )
        
        print("pin_hole_mask\n",pin_hole_mask)
    return pin_hole_mask


def draw_line_upwards_from_point(
    mask: np.ndarray, start, fill_value: int
) -> np.ndarray:
    line_width = 1 #pas bien compris l'impact de ce truc mais avant a 2
    end = (start[0], start[1] - 1)
    print("end",end)
    print("fill_value",fill_value)
    mask = mask.astype(np.uint8)  # Type that OpenCV expects
    # Draw one point at a time until we hit a point that already has the desired value
    while mask[end] != fill_value:
        cv.line(mask, start, end, fill_value, line_width)

        # Update start and end to the next positions
        start = end
        end = (start[0], start[1] - line_width)
    return mask.astype(bool)


def validate_contours(contours: list):
    if len(contours) == 0:
        raise Exception(
            "Unable to find contour in non empty mask, please check your mask formatting"
        )


def get_pixel_to_patient_transformation_matrix(series_data):
    """
    https://nipy.org/nibabel/dicom/dicom_orientation.html
    """

    first_slice = series_data[0]

    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)

    mat = np.identity(4, dtype=np.float32)
    mat[:3, 0] = row_direction * row_spacing
    mat[:3, 1] = column_direction * column_spacing
    mat[:3, 2] = slice_direction * slice_spacing
    mat[:3, 3] = offset

    return mat


def get_patient_to_pixel_transformation_matrix(series_data):
    first_slice = series_data[0]

    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)

    # M = [ rotation&scaling   translation ]
    #     [        0                1      ]
    #
    # inv(M) = [ inv(rotation&scaling)   -inv(rotation&scaling) * translation ]
    #          [          0                                1                  ]

    linear = np.identity(3, dtype=np.float32)
    linear[0, :3] = row_direction / row_spacing
    linear[1, :3] = column_direction / column_spacing
    linear[2, :3] = slice_direction / slice_spacing

    mat = np.identity(4, dtype=np.float32)
    mat[:3, :3] = linear
    mat[:3, 3] = offset.dot(-linear.T)

    return mat


def apply_transformation_to_3d_points(
    points: np.ndarray, transformation_matrix: np.ndarray
):
    """
    * Augment each point with a '1' as the fourth coordinate to allow translation
    * Multiply by a 4x4 transformation matrix
    * Throw away added '1's
    """
    vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return vec.dot(transformation_matrix.T)[:, :3]


def get_slice_position(series_slice: Dataset):
    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ImagePositionPatient)


def get_slice_directions(series_slice: Dataset):
    orientation = series_slice.ImageOrientationPatient
    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)

    if not np.allclose(
        np.dot(row_direction, column_direction), 0.0, atol=1e-3
    ) or not np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3):
        raise Exception("Invalid Image Orientation (Patient) attribute")

    return row_direction, column_direction, slice_direction


def get_spacing_between_slices(series_data):
    if len(series_data) > 1:
        first = get_slice_position(series_data[0])
        last = get_slice_position(series_data[-1])
        return (last - first) / (len(series_data) - 1)

    # Return nonzero value for one slice just to make the transformation matrix invertible
    return 1.0


def create_series_mask_from_contour_sequence(series_data, contour_sequence: Sequence):
    mask = create_empty_series_mask(series_data)
    transformation_matrix = get_patient_to_pixel_transformation_matrix(series_data)

    # Iterate through each slice of the series, If it is a part of the contour, add the contour mask
    for i, series_slice in enumerate(series_data):
        slice_contour_data = get_slice_contour_data(series_slice, contour_sequence)
        if len(slice_contour_data):
            mask[:, :, i] = get_slice_mask_from_slice_contour_data(
                series_slice, slice_contour_data, transformation_matrix
            )
    return mask


def get_slice_contour_data(series_slice: Dataset, contour_sequence: Sequence):
    slice_contour_data = []

    # Traverse through sequence data and get all contour data pertaining to the given slice
    for contour in contour_sequence:
        for contour_image in contour.ContourImageSequence:
            if contour_image.ReferencedSOPInstanceUID == series_slice.SOPInstanceUID:
                slice_contour_data.append(contour.ContourData)

    return slice_contour_data


def get_slice_mask_from_slice_contour_data(
    series_slice: Dataset, slice_contour_data, transformation_matrix: np.ndarray
):
    slice_mask = create_empty_slice_mask(series_slice)
    for contour_coords in slice_contour_data:
        fill_mask = get_contour_fill_mask(
            series_slice, contour_coords, transformation_matrix
        )
        # Invert values in the region to be filled. This will create holes where needed if contours are stacked on top of each other
        slice_mask[fill_mask == 1] = np.invert(slice_mask[fill_mask == 1])
    return slice_mask


def get_contour_fill_mask(
    series_slice: Dataset, contour_coords, transformation_matrix: np.ndarray
):
    # Format data
    reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
    translated_contour_data = apply_transformation_to_3d_points(
        reshaped_contour_data, transformation_matrix
    )
    polygon = [np.around([translated_contour_data[:, :2]]).astype(np.int32)]

    # Create mask for the region. Fill with 1 for ROI
    fill_mask = create_empty_slice_mask(series_slice).astype(np.uint8)
    cv.fillPoly(img=fill_mask, pts=polygon, color=1)
    return fill_mask


def create_empty_series_mask(series_data):
    ref_dicom_image = series_data[0]
    mask_dims = (
        int(ref_dicom_image.Columns),
        int(ref_dicom_image.Rows),
        len(series_data),
    )
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def create_empty_slice_mask(series_slice):
    mask_dims = (int(series_slice.Columns), int(series_slice.Rows))
    mask = np.zeros(mask_dims).astype(bool)
    return mask


class Hierarchy(IntEnum):
    """
    Enum class for what the positions in the OpenCV hierarchy array mean
    """

    next_node = 0
    previous_node = 1
    first_child = 2
    parent_node = 3

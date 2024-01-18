#ifndef  _MMVII_Serial_Tpl2Inc_H_
#define  _MMVII_Serial_Tpl2Inc_H_

/** \file MMVII_2Include_Serial_Tpl.h
    \brief Contains template definition  for serialisation

   This file must be include when instatiation serialization
 on Type constructed on other like "Pointeur on Type", "Vector of Type" ...

*/

#include "MMVII_DeclareCste.h"
#include "MMVII_Stringifier.h"
#include "MMVII_Matrix.h"
#include "MMVII_2Include_CSV_Serial_Tpl.h"
#include <set>
#include <map>


namespace MMVII
{
   
void PushPrecTxtSerial(size_t aPrec);  /// set new precision for txt-serialisation
void PopPrecTxtSerial();   /// restore precision for txt-serialisation




typedef  std::map<std::string,std::vector<cPt2dr>>  tTestMasSerial;  /// Type for basic test-serialisation of maps
tTestMasSerial  GentTestMasSerial(); /// Generate a sample for test



template <class Type> void TplAddRawData(const cAuxAr2007 & anAux,Type * anAdr,int aNbElem,const std::string & aTag="RawData")
{
    cRawData4Serial aRDS = cRawData4Serial::Tpl(anAdr,aNbElem);
    AddData(cAuxAr2007(aTag,anAux),aRDS);
}

template <class Type> void EnumAddData(const cAuxAr2007 & anAux,Type & anEnum,const std::string & aTag)
{
   if (anAux.Tagged())
   {
       // modif MPD , if input enum is not init
       std::string aName =   (anAux.Ar().IsSpecif())  ?
	                       ("enum_"+ cStrIO<Type>::msNameType)                   :  // Not sure what to put in case of specification file
	                       (anAux.Input() ? std::string("") :E2Str(anEnum) ) ;
       AddData(cAuxAr2007(aTag,anAux),aName);
       if (anAux.Input())
          anEnum = Str2E<Type>(aName);
   }
   else
   {
       int aIEnum = int(anEnum);
       AddData(cAuxAr2007(aTag,anAux),aIEnum);
       if (anAux.Input())
          anEnum =  (Type) aIEnum;
   }
}


/// Serialization for optional
/** Template for optional parameter, complicated becaus in xml forms, 
    it handles the compatibility with new added parameters 
 
    Name it AddOptData and not  AddData, because on this experimental stuff,
    want do get easy track of it.

*/

template <class Type> void AddOptData(const cAuxAr2007 & anAux,const std::string & aTag0,std::optional<Type> & aL)
{
    // put the tag as <Opt::Tag0>,
    //  Not mandatory, but optionality being an important feature I thought usefull to see it in XML file
    //  put it
    std::string aTagOpt;
    const std::string * anAdrTag = & aTag0;
    if (anAux.Tagged())
    {
        aTagOpt = "Opt:" + aTag0;
        anAdrTag = & aTagOpt;
    }

   // In input mode, we must decide if the value is present
    if (anAux.Input())
    {
        // The archive knows if the object is present
        if (anAux.NbNextOptionnal(*anAdrTag))
        {
           // If yes read it and initialize optional value
           Type  aV;
           AddData(cAuxAr2007(*anAdrTag,anAux),aV);
           aL = aV;
        }
        // If no just put it initilized
        else
           aL = std::nullopt;
        return;
    }

    // Now in writing mode
    int aNb =  aL.has_value() ? 1 : 0;
    // Tagged format (xml) is a special case
    if (anAux.Tagged())
    {
       // If the value exist put it normally else do nothing (the absence of tag will be analysed at reading)
       if (aNb)
          AddData(cAuxAr2007(*anAdrTag,anAux),*aL);
    }
    else
    {
       // Indicate if the value is present and if yes put it
       AddData(anAux,aNb);  
       anAux.Ar().Separator();
       if (aNb)
          AddData(anAux,*aL);
    }
}

template <class Type, size_t size> void AddOptTabData(const cAuxAr2007 & anAux,const std::string & aTag0,std::optional<cArray<Type, size>> & aL)
{
    // put the tag as <Opt::Tag0>,
    //  Not mandatory, but optionality being an important feature I thought usefull to see it in XML file
    //  put it
    std::string aTagOpt;
    const std::string * anAdrTag = & aTag0;
    if (anAux.Tagged())
    {
        aTagOpt = "Opt:" + aTag0;
        anAdrTag = & aTagOpt;
    }

   // In input mode, we must decide if the value is present
    if (anAux.Input())
    {
        // The archive knows if the object is present
        if (anAux.NbNextOptionnal(*anAdrTag))
        {
           // If yes read it and initialize optional value
           cArray<Type, size> aV;
           AddTabData(cAuxAr2007(*anAdrTag,anAux),aV.data(), size);
           aL = aV;
        }
        // If no just put it initilized
        else
           aL = std::nullopt;
        return;
    }

    // Now in writing mode
    int aNb =  aL.has_value() ? 1 : 0;
    // Tagged format (xml) is a special case
    if (anAux.Tagged())
    {
       // If the value exist put it normally else do nothing (the absence of tag will be analysed at reading)
       if (aNb)
          AddTabData(cAuxAr2007(*anAdrTag,anAux),aL->data(), size);
    }
    else
    {
       // Indicate if the value is present and if yes put it
       AddData(anAux,aNb);
       anAux.Ar().Separator();
       if (aNb)
          AddTabData(anAux,aL->data(), size);
    }
}


/// Pointer serialisation, make the assumption that pointer are valide (i.e null or dynamically allocated)
template <class Type> void OnePtrAddData(const cAuxAr2007 & anAux,Type * & aL)
{
     bool doArchPtrNull = (aL==nullptr);
     if (anAux.Tagged())
     {
        // This case probably tricky to support correctly, 4 now generate a error, will see later if 
        // support is required
        MMVII_INTERNAL_ASSERT_strong(doArchPtrNull,"AddData Null Ptr in Xml-file not supported for now");
     }
     else
     {
         AddData(anAux,doArchPtrNull);
     }
     // If pointer 0, no write/no read
     if (doArchPtrNull)
     {
         if (anAux.Input())  // seems logical to reput the null ptr
         {
             delete aL;
             aL = nullptr;
         }
         return;
     }
     // In read mode, if the Non nul where archived and aL is currently null need to allocate room for value
     if (anAux.Input() && (aL==nullptr))
     {
         aL = new Type;
     }
/*
*/
     AddData(anAux,*aL);
}

// need general inteface for things like std::vector<Type *>
template <class Type> void AddData(const cAuxAr2007 & anAux,Type * & aL)
{
    OnePtrAddData(anAux,aL);
}

/// Const Pointer serialisation
template <class Type> void AddData(const cAuxAr2007 & anAux,const Type * & aL)
{
     AddData(anAux,const_cast<Type*&>(aL));
     // AddData(anAux,const_cast<Type&>(*aL));
}

extern void AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux);


/// Serialization for stl container
/** Thi should work both for stl containers (require size + iterator auto)
*/

extern const std::string  StrElCont;
extern const std::string  StrElMap;


template <class TypeCont> void StdContAddData(const cAuxAr2007 & anAux,TypeCont & aL)
{
    anAux.SetType(eTAAr::eCont);
    int aNb=aL.size();
    if (anAux.Ar().IsSpecif())
       aNb = 1;
    // put or read the number
    // AddData(cAuxAr2007("Nb",anAux),aNb);
    AddDataSizeCont(aNb,anAux);
    // In input, nb is now intialized, we must set the size of list
    if (aNb!=int(aL.size()))
    {  
       //typename TypeCont::value_type aV0 ;
       //aL = TypeCont(aNb,aV0);
       aL = TypeCont(aNb);
    }
    // now read the elements
    for (auto & el : aL)
    {    
         AddData(cAuxAr2007(StrElCont,anAux,eTAAr::eElemCont),el);
    }
}

/// std::list interface  AddData -> StdContAddData
template <class Type> void AddData(const cAuxAr2007 & anAux,std::list<Type>   & aL) { StdContAddData(anAux,aL); }
/// std::vector interface  AddData -> StdContAddData
template <class Type> void AddData(const cAuxAr2007 & anAux,std::vector<Type> & aL) { StdContAddData(anAux,aL); }


/** Serialization for map (will be) used for cSetMultipleTiePoints, and more ? */

template <class TypeKey,class TypeVal> void AddData(const cAuxAr2007 & anAux,std::map<TypeKey,TypeVal> & aMap)
{
    anAux.SetType(eTAAr::eMap);
    int aNb=aMap.size();
    // put or read the number
    //  AddData(cAuxAr2007("Nb",anAux),aNb);
    AddDataSizeCont(aNb,anAux);
    // a bit trick the iteration is fundamentally different in input and output, because can't easily
    // fix the size
    if (anAux.Input())
    {
       // when read parse the Number of pair, read the key and put the value in the key
       for (int aK=0 ; aK<aNb ; aK++)
       {
          {
            cAuxAr2007 anAuxPair(StrElMap,anAux,eTAAr::ePairMap);
            TypeKey aKey;
            AddData(cAuxAr2007("K",anAuxPair,eTAAr::eKeyMap),aKey);
            AddData(cAuxAr2007("V",anAuxPair,eTAAr::eValMap),aMap[aKey]);
          }
       }
    }
    else
    {
        if (anAux.Ar().IsSpecif())
        {
            aMap.clear();
            // aMap[TypeKey{}] = TypeVal{};
            aMap.try_emplace(TypeKey{}); // TypeVal{};
        }
       // when write parse the map,
        for (auto & aPair : aMap)
        {
            cAuxAr2007 anAuxPair(StrElMap,anAux,eTAAr::ePairMap);
            AddData(cAuxAr2007("K",anAuxPair,eTAAr::eKeyMap),const_cast<TypeKey&>(aPair.first));
            AddData(cAuxAr2007("V",anAuxPair,eTAAr::eValMap),const_cast<TypeVal&>(aPair.second));
            //AddData(anAuxPair,aPair->second);
        }
    }
}



template <class Type,const int Dim> void AddData(const cAuxAr2007 & anAux,cDataTypedIm<Type,Dim> & aIm)
{
    cPtxd<int,Dim> aSz = aIm.Sz();
    AddData(cAuxAr2007("Sz",anAux),aSz);
    if (anAux.Input())
    { 
      aIm.Resize(cPtxd<int,Dim>::PCste(0),aSz);
    }

    TplAddRawData(anAux,aIm.RawDataLin(),aIm.NbElem());
/*
    cRawData4Serial aRDS = cRawData4Serial::Tpl(aIm.RawDataLin(),aIm.NbElem());
    AddData(cAuxAr2007("Data",anAux),aRDS);
*/
}

template <class Type> void AddData(const cAuxAr2007 & anAux, cDenseVect<Type>& aVect)
{
    AddData(anAux,aVect.DIm());
}

// template <class Type,const int Dim> void AddData(const cAuxAr2007 & anAux,cDataTypedIm<Type,Dim> & aIm)


template <class TypeH,class TypeCumul> void AddData(const cAuxAr2007 & anAux,cHistoCumul<TypeH,TypeCumul> & aHC)
{
    aHC.AddData(anAux);
}

template <class Type> void AddData(const cAuxAr2007 & anAux, cDataGenDimTypedIm<Type> & aImND)
{
    aImND.AddData(anAux);
}


/// cExtSet  serialisation
template <class Type> void AddData(const cAuxAr2007 & anAux,cExtSet<Type> & aSet)
{
    cAuxAr2007 aTagSet(XMLTagSet<Type>(),anAux);
    if (anAux.Input())  // If we are reading the "file"
    {
        std::vector<Type> aV; // read data in a vect
        AddData(aTagSet,aV);
        for (const auto & el: aV)  // put the vect in the set G++11
            aSet.Add(el);
    }
    else
    {
        // A bit un-opt, because we create copy, but it would be dangerous (and lead to real error ...)
        // to write as pointer and read as object
        std::vector<const Type *> aVPtr;  // put the set in a vect
        aSet.PutInVect(aVPtr,true);
        std::vector<Type> aVObj;  // make a copy as object
        for (auto aPtr : aVPtr)
            aVObj.push_back(*aPtr);
        AddData(aTagSet,aVObj);  // "write" the vect
    }
}


/// Save the value in an archive, not proud of the const_cast ;-)
/**  SaveInFile :
     Handle the V1/V2 choice
     Allocate the archive from name (Xml, binary, ...)
     Write using AddData
*/
template<class TypeVal> void  TopAddAr(cAr2007  & anAr,TypeVal & aVal,const std::string & aName)
{
    std::string aStrVersion ="0.0.0";
    std::string aStrSerial  =TagMMVIISerial;
    std::string aStrRoot    =TagMMVIIRoot;

    std::string aLP =  LastPostfix(aName);
    bool IsXml =  (aLP=="xml") || (aLP=="xml2");


    if (IsXml)
    {
       cAuxAr2007  aG0(aStrRoot,anAr,eTAAr::eStd);
       // AddData(cAuxAr2007("Type"   ,aG0,eTAAr::eStd),aVS);
       AddData(cAuxAr2007(TagMMVIIType    ,aG0,eTAAr::eStd),aStrSerial);
       AddData(cAuxAr2007(TagMMVIIVersion ,aG0,eTAAr::eStd),aStrVersion);
       AddData(cAuxAr2007(TagMMVIIData    ,aG0,eTAAr::eStd),aVal);
    }
    else
    {
       AddData(cAuxAr2007(TagMMVIIType   ,anAr,eTAAr::eStd),aStrSerial);
       AddData(cAuxAr2007(TagMMVIIVersion,anAr,eTAAr::eStd),aStrVersion);
       AddData(cAuxAr2007(TagMMVIIData,   anAr,eTAAr::eStd),aVal);
    }
}

template<class Type> void  GenSaveInFile(const Type & aVal,const std::string & aName,bool IsSpecif)
{
   if (GlobOutV2Format())  // Do we save using MMV2 format by serialization
   {
       // Unique Ptr  , second type indicate the type of deleting unction
       std::unique_ptr<cAr2007>  anAr (AllocArFromFile(aName,false,IsSpecif));

           /// Not proud of cons_cast ;-( 
       TopAddAr(*anAr,const_cast<Type&>(aVal),aName);
   }
   else
   {
     MMv1_SaveInFile<Type>(aVal,aName);
   }
}
template<class Type> void  SaveInFile_Std(const Type & aVal,const std::string & aName)
{
	GenSaveInFile(aVal,aName,false);
}

///

template<class Type> void  SaveInFile(const std::vector<Type> & aVec,const std::string & aName)
{
    if (LastPostfix(aName) == E2Str(eTypeSerial::ecsv))
    {
        ToCSV(aVec,aName,true);
    }
    else
    {
        SaveInFile_Std(aVec,aName);
    }
}

template<class Type> void  SaveInFile(const Type & aVal,const std::string & aName)
{
	 SaveInFile_Std(aVal,aName);
}


template<class Type> void  SpecificationSaveInFile(const std::string & aName)
{
     Type aVal;
     GenSaveInFile(aVal,aName,true);
}

template<class Type> void  SpecificationSaveInFile()
{
     SpecificationSaveInFile<Type>("Specifications_"+cStrIO<Type>::msNameType+"."+GlobTaggedNameDefSerial());
}


template<class Type> size_t  HashValue(cAr2007 * anAr,const Type & aVal,bool ordered)
{
    cAuxAr2007  aGLOB(TagMMVIISerial,*anAr, eTAAr::eStd);
    AddData(aGLOB,const_cast<Type&>(aVal));
    return HashValFromAr(*anAr);
}
template<class Type> size_t  HashValue(const Type & aVal,bool ordered)
{
    std::unique_ptr<cAr2007>  anAr (AllocArHashVal(ordered));
    return HashValue(anAr.get(),aVal,ordered);
}




/** Same as write, but simpler as V1/V2 choice is guided by file */
template<class Type> void  ReadFromFile_Std(Type & aVal,const std::string & aName)
{
    std::unique_ptr<cAr2007>  anAr (AllocArFromFile(aName,true));
    TopAddAr(*anAr,aVal,aName);
}

template<class Type> void  ReadFromFile(std::vector<Type> & aVec,const std::string & aName)
{
    if (LastPostfix(aName) == E2Str(eTypeSerial::ecsv))
    {
        FromCSV(aVec,aName,true);
    }
    else
    {
        ReadFromFile_Std(aVec,aName);
    }
}

template<class Type> void  ReadFromFile(Type & aVal,const std::string & aName)
{
    ReadFromFile_Std(aVal,aName);
}



/// If the file does not exist, initialize with default constructor
template<class Type> void  ReadFromFileWithDef(Type & aVal,const std::string & aName)
{
   if (ExistFile(aName))
      ReadFromFile(aVal,aName);
   else
      aVal = Type();
}

///  Save in file if it's the first times it occurs inside the process
template<class Type> void  ToFileIfFirstime(const Type * anObj,const std::string & aNameFile,bool ForReset=false)
{
   static std::set<std::string> aSetFilesAlreadySaved;
   if (ForReset)
   {
       aSetFilesAlreadySaved.clear();
       return;
   }

   if (!BoolFind(aSetFilesAlreadySaved,aNameFile))
   {
        aSetFilesAlreadySaved.insert(aNameFile);
        anObj->ToFile(aNameFile);
   }
}
template<class Type> void  ResetToFileIfFirstime()
{
    ToFileIfFirstime((Type*)nullptr,"",true);
}



template<class Type,class TypeTmp> Type * ObjectFromFile(const std::string & aName)
{
    TypeTmp aDataCreate;
    ReadFromFile(aDataCreate,aName);
    return new Type(aDataCreate);
}

/**  Read in the file if first time and memorize, other times return the same object ,
 *   at end, destruction will be handled using "AddObj2DelAtEnd"  (which is required for memory checking)
 */
template<class Type,class TypeTmp> Type * RemanentObjectFromFile(const std::string & aName)
{
     static std::map<std::string,Type *> TheMap;
     Type * & anExistingRes = TheMap[aName];

     if (anExistingRes == 0)
     {
        // TypeTmp aDataCreate;
        // ReadFromFile(aDataCreate,aName);
        // anExistingRes = new Type(aDataCreate);
        anExistingRes = ObjectFromFile<Type,TypeTmp>(aName);
        cMMVII_Appli::AddObj2DelAtEnd(anExistingRes);
     }
     return anExistingRes;
}

};

#endif //  _MMVII_Serial_Tpl2Inc_H_


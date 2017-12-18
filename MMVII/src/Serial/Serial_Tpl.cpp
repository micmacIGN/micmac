#include "include/MMVII_all.h"
#include "include/MMVII_Class4Bench.h"



/** \file Serial_Tpl.cpp
    \brief Serialization explicit instatiation


*/


namespace MMVII
{

// Pointeur
template <class Type> void AddData(const cAuxAr2007 & anAux,Type * aL)
{
     AddData(anAux,*aL);
}
template <class Type> void AddData(const cAuxAr2007 & anAux,const Type * aL)
{
     AddData(anAux,const_cast<Type&>(*aL));
}


template  void AddData(const cAuxAr2007 & anAux,std::string * aL);
template  void AddData(const cAuxAr2007 & anAux,const std::string * aL);


// ExtSet

template <class Type> void AddData(const cAuxAr2007 & anAux,cExtSet<Type> & aSet)
{
    cAuxAr2007 aTagSet(XMLTagSet<Type>(),anAux);
    if (anAux.Input())
    {
        std::vector<Type> aV;
        AddData(aTagSet,aV);
        for (const auto el: aV)
            aSet.Add(el);
    }
    else
    {
        std::vector<const Type *> aV;
        aSet.PutInVect(aV,true);
        AddData(aTagSet,aV);
    }
}

template void AddData(const cAuxAr2007 & anAux,cExtSet<std::string> & aSet);
template void AddData(const cAuxAr2007 & anAux,cExtSet<tNamePair> & aSet);

// Optionnal


/// Serialization for optional
/** Template for optional parameter, complicated becaus in xml forms, 
    it handles the compatibility with new added parameters 
 
    Name it AddOptData and not  AddData, because on this experimental stuff,
    want do get easy track of it.

*/

template <class Type> void AddOptData(const cAuxAr2007 & anAux,const std::string & aTag0,boost::optional<Type> & aL)
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
           aL = boost::none;
        return;
    }

    // Now in writing mode
    int aNb =  aL.is_initialized() ? 1 : 0;
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
       if (aNb)
          AddData(anAux,*aL);
    }
}

template void AddOptData(const cAuxAr2007 & anAux,const std::string & aTag0,boost::optional<cPt2dr> & aL);


template <class TypeCont> void StdContAddData(const cAuxAr2007 & anAux,TypeCont & aL)
{
    int aNb=aL.size();
    // put or read the number
    AddData(cAuxAr2007("Nb",anAux),aNb);
    // In input, nb is now intialized, we must set the size of list
    if (aNb!=int(aL.size()))
    {  
       typename TypeCont::value_type aV0;
       aL = TypeCont(aNb,aV0);
    }
    // now read the elements
    for (auto & el : aL)
    {    
         AddData(cAuxAr2007("el",anAux),el);
    }
}

template  void StdContAddData(const cAuxAr2007 & anAux,std::list<int> & aL);
template  void StdContAddData(const cAuxAr2007 & anAux,std::vector<double> & aL);
template  void StdContAddData(const cAuxAr2007 & anAux,std::vector<std::string> & aL);




};


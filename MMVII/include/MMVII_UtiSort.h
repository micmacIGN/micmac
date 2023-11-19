#ifndef  _MMVII_UTI_SORT_H_
#define  _MMVII_UTI_SORT_H_

namespace MMVII
{
/**  Methof for sorting simultaneously 2 vector uing lexicographic order => the 2 type must be comparable
      => maybe in future moe=re efficient implementation using a permutation ?
*/

template <class T1,class T2>  void Sort2VectLexico(std::vector<T1> &aV1,std::vector<T2> & aV2)
{
     MMVII_INTERNAL_ASSERT_tiny(aV1.size()==aV2.size(),"Diff size in Sort2V");

     std::vector<std::pair<T1,T2> > aV12;

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
         aV12.push_back(std::pair<T1,T2>(aV1.at(aK),aV2.at(aK)));

     std::sort(aV12.begin(),aV12.end());
     // std::sort(aV12.begin(),aV12.end(),[](const auto &aP1,const auto & aP2){return aP1.first < aP2.first;});

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
     {
          aV1.at(aK) = aV12.at(aK).first;
          aV2.at(aK) = aV12.at(aK).second;
     }
}

/**  Methof for sorting simultaneously 2 vector using only first vector; only first type must be comparible,
       and order among equivalent first value is undefined
      => maybe in future moe=re efficient implementation using a permutation ?
*/

template <class T1,class T2>  void Sort2VectFirstOne(std::vector<T1> &aV1,std::vector<T2> & aV2)
{
     MMVII_INTERNAL_ASSERT_tiny(aV1.size()==aV2.size(),"Diff size in Sort2V");

     std::vector<std::pair<T1,T2> > aV12;

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
         aV12.push_back(std::pair<T1,T2>(aV1.at(aK),aV2.at(aK)));

     std::sort(aV12.begin(),aV12.end(),[](const auto &aP1,const auto & aP2){return aP1.first < aP2.first;});

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
     {
          aV1.at(aK) = aV12.at(aK).first;
          aV2.at(aK) = aV12.at(aK).second;
     }
}


};

#endif  //  _MMVII_UTI_SORT_H_

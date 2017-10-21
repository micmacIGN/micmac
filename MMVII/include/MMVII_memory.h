#ifndef  _MMVII_MEMORY_H_
#define  _MMVII_MEMORY_H_


void mem_raz(void * adr,int64_t nb);
#define MEM_RAZ(x,nb) mem_raz((void *)(x),(nb)*sizeof(*(x)))


class  cMemState;
class  cMemManager;
class  cMemCheck;

// Classe permettant de traquer les principales erreurs memoire


class cMemState
{
    public :
        friend class cMemManager;
        bool operator == (const cMemState &) const;
        int NbObjCreated() const;
    private :
        cMemState();
        int64_t  mCheckNb;
        int64_t  mCheckSize;
        int64_t  mCheckPtr;
        int64_t  mNbObjCreated;
};


class cMemManager
{
    public :
        // si debug assez eleve , memorize les alloc/desalloc 
        // pour pouvoir verifier si  tout ce qui est alloue est rendu 
        // rajoute un majic number pour verifier que l'on n'a pas ecrit en bordure
        static void * Calloc(size_t nmemb, size_t size);
        static void   Free(void * ptr);

        // Permet d'enregistrer un etat memoire puis de verifier
        // a la fin d'une execution, que tout a ete rendu
        static const cMemState  CurState() ;
        static void  CheckRestoration(const cMemState &) ;
        static bool  IsOkCheckRestoration(const cMemState &) ;

        // Par ex :    short * aPtr = cMemManager::Alloc<short>(aNb);
        template <class Type> static inline Type * Alloc(size_t nmemb)
        {
           return  static_cast<Type *> (cMemManager::Calloc(nmemb,sizeof(Type)));
        }
    private :

        static cMemState mState;
};


// Ces classe redefini l'operetur new pour checker alloc / desalloc et debordement

class  cMemCheck
{
      public :
         void * operator new    (size_t sz);
         void operator delete   (void * ptr) ;

      private :

       // to avoid use 
         void * operator new []  (size_t sz);
         void operator delete [] (void * ptr) ;
};



//  Utilitaire courant 
template <class Type> inline void DeleteAllAndClear(Type & aVal)
{
    for (auto it=aVal.begin() ; it!=aVal.end() ; it++)
        delete *it;
    aVal.clear();
}

template<class Type> class cGestObjetEmpruntable
{
     public :
         cGestObjetEmpruntable() :
               mNbEmprunte (0)
         {
         }
         ~cGestObjetEmpruntable()
         {
             MMVII_INTERNAL_ASSERT_medium(mNbEmprunte==0,"Bad handling of cGestObjetEmpruntable");
             DeleteAllAndClear(mReserve);
         }

         Type * EmprunterOne()
         {
             mNbEmprunte++;
             if (!mReserve.empty())
             {
                  Type * aRes = mReserve.back();
                  mReserve.pop_back();
                  return aRes;
             }
             return new Type;
         }
         void RendreOne(Type * aVal)
         {
             mNbEmprunte--;
             mReserve.push_back(aVal);
         }
     private :
         int                 mNbEmprunte;
         std::vector<Type *> mReserve;
};





#endif  //  _MMVII_MEMORY_H_

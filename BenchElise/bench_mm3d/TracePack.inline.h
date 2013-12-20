// this file is supposed to be included only in TracePack.h

//--------------------------------------------
// class TracePack::Registry::Item
//--------------------------------------------

TracePack::Registry::Item::Item( const cElFilename &i_filename,
	                         TD_Type i_type,
	                         const cElDate &i_date,
	                         streampos i_dataOffset,
	                         unsigned int i_dataSize ):
   m_filename( i_filename ),
   m_type( i_type ),
   m_date( i_date ),
   m_dataOffset( i_dataOffset ),
   m_dataSize( i_dataSize )
{
}


//--------------------------------------------
// class TracePack::Registry
//--------------------------------------------

size_t TracePack::Registry::size() const { return m_items.size(); }


//--------------------------------------------
// class TracePack
//--------------------------------------------

unsigned int TracePack::nbStates() const { return m_registries.size(); }

const cElDate & TracePack::date() const { return m_date; }

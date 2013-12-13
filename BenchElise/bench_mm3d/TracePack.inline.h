// this file is supposed to be included only in TracePack.h


//--------------------------------------------
// class TracePack::Registry
//--------------------------------------------

size_t TracePack::Registry::size() const { return m_items.size(); }


//--------------------------------------------
// class TracePack
//--------------------------------------------

unsigned int TracePack::nbStates() const { return m_registries.size(); }

const cElFilename & TracePack::filename() const { return m_filename; }

const cElDate & TracePack::date() const { return m_date; }

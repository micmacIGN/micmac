#include "MessageHandler.h"

using namespace std;

//----------------------------------------------------------------------
// methods of class ListMessageHandler
//----------------------------------------------------------------------

ListMessageHandler::~ListMessageHandler()
{
	for ( list<Message*>::iterator it=m_messages.begin(); it!=m_messages.end(); it++ )
		delete *it;
}

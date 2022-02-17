#ifndef __MESSAGE_HANDLER__
#define __MESSAGE_HANDLER__

#include <iostream>
#include <string>
#include <list>
#include <cstdlib>
#include <sstream>

class Message
{
public:
	virtual Message * duplicate() const = 0;
	virtual std::string toString() const = 0;
	virtual ~Message();
};

class DebugErrorMessage : public Message
{
public:
	std::string m_file;
	int         m_line;
	std::string m_where;
	std::string m_what;

	DebugErrorMessage( const std::string &i_file, int i_line, const std::string &i_where, const std::string &i_what );

	// Message interface
	Message * duplicate() const;
	std::string toString() const;
};

class StringMessage : public Message
{
public:
	std::string m_value;

	StringMessage( const std::string &i_value );

	// Message interface
	Message * duplicate() const;
	std::string toString() const;
};

class MessageHandler
{
public:
	typedef enum
	{
		CIN_GET, EXIT, NOTHING
	} eAction;

protected:
	eAction m_action;
	int     m_exitCode;

public:
	MessageHandler( eAction i_action=NOTHING, int i_exitCode=0 );
	virtual void add( const Message &i_message ) = 0;
	virtual ~MessageHandler();
	void setAction( eAction i_action, int i_exitCode=0 );
	eAction action() const;
	int exitCode() const;
};

class ListMessageHandler : public MessageHandler
{
private:
	std::list<Message *> m_messages;

public:
	ListMessageHandler();
	~ListMessageHandler();

	// MessageHandler interface
	void add( const Message &i_message );
};

class OStreamMessageHandler : public MessageHandler
{
private:
	std::ostream & m_stream;
	std::string    m_prefix;

public:
	OStreamMessageHandler( std::ostream &io_ostream, const std::string &i_prefix, eAction i_action, int i_exitCode );
	~OStreamMessageHandler();

	// MessageHandler interface
	void add( const Message &i_message );
};

#define STR_MESSAGE(handler,msg) {\
	stringstream ss;\
	ss << msg;\
	handler.add( StringMessage( ss.str() ) );\
}

std::string eToString(MessageHandler::eAction e);

#include "MessageHandler.inline.h"

#endif

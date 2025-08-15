#pragma once

#include "writer.hpp"

#include <memory>
#include <ostream>


class CoutWriter : public Writer
{
public:
    CoutWriter() : Writer()
    {
        m_output.reset(&std::cout);
    }
    ~CoutWriter()
    {
        m_output.release();
    }
};

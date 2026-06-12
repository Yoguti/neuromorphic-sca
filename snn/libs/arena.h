/* dev/arena.h - Simple memory arena allocator
 *
 * This header defines a simple memory arena allocator for efficient
 * memory management in C programs. It allows for fast allocation
 * and deallocation of memory blocks.
 *
 * Usage:
 * 1. Inside one C file, define ARENA_IMPLEMENTATION before including this
 * header #define ARENA_IMPLEMENTATION #include "dev/arena.h"
 * 2. In other files, simply include the header
 * #include "dev/arena.h"
 * 3. Use the provided functions to manage memory arenas.
 * Author: Klaus Schneider
 *
 */

#ifndef DEV_ARENA_H
#define DEV_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Arena
{
  struct Arena *next;
  size_t capacity;
  size_t offset;
  unsigned char buffer[];
} Arena;

Arena *arena_init(size_t capacity);
void  *arena_alloc(Arena *arena, size_t size);
void   arena_free(Arena *arena);
void   arena_reset(Arena *arena);

#define arena_push_struct(arena, type) (type *)arena_alloc(arena, sizeof(type))

#define arena_push_array(arena, type, count) \
  (type *)arena_alloc(arena, sizeof(type) * (count))

#endif /* DEV_ARENA_H */

#ifdef ARENA_IMPLEMENTATION

Arena *arena_init(size_t capacity)
{
  Arena *arena = (Arena *)malloc(sizeof(*arena) + capacity);
  if (arena == NULL)
  {
    return NULL;
  }
  arena->next = NULL;
  arena->capacity = capacity;
  arena->offset = 0;
  return arena;
}

void *arena_alloc(Arena *arena, size_t size)
{
  size_t align    = _Alignof(max_align_t);
  uintptr_t base  = (uintptr_t)&arena->buffer[arena->offset];
  size_t   padding = (align - (base % align)) % align;
  size_t   aligned = arena->offset + padding;

  if (aligned + size > arena->capacity)
  {
    if (arena->next == NULL)
    {
      size_t next_cap = arena->capacity;
      if (size + align > next_cap) next_cap = size + align;
      arena->next = arena_init(next_cap);
      if (arena->next == NULL)
      {
        return NULL;
      }
    }
    return arena_alloc(arena->next, size);
  }

  arena->offset = aligned + size;
  return &arena->buffer[aligned];
}

void arena_free(Arena *arena)
{
  while (arena != NULL)
  {
    Arena *next = arena->next;
    free(arena);
    arena = next;
  }
}

void arena_reset(Arena *arena)
{
  while (arena != NULL)
  {
    arena->offset = 0;
    arena = arena->next;
  }
}

#endif /* ARENA_IMPLEMENTATION */
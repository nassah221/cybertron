// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const limit = 10

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVar("CYBERTRON_MODELS_DIR")
	modelName := HasEnvVar("CYBERTRON_MODEL")

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	defer tasks.Finalize(m)

	r1, err := m.Encode(context.Background(), "This is a happy person", int(bert.MeanPooling))
	if err != nil {
		panic(err)
	}

	r2, err := m.Encode(context.Background(), "The cat is purring on my lap", int(bert.MeanPooling))
	if err != nil {
		panic(err)
	}

	fmt.Println(r2.Vector.Normalize2().DotUnitary(r1.Vector.Normalize2()))
}

func dotProduct[T []float64](a, b T) float64 {
	result := 0.0
	if len(a) != len(b) {
		panic("vector length not equal")
	}
	for i := 0; i < len(a); i++ {
		result += a[i] * b[i]
	}
	return result
}

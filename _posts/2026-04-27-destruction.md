---
layout: distill
title: Destruction is a General Strategy to Learn Generation; Diffusion's Strength is to Take it Seriously; Exploration is the Future
description: I present diffusion models as part of a family of machine learning techniques that withhold information from a model’s input and train it to guess the withheld information. I argue that diffusion's destroying approach to withholding is more flexible than typical hand-crafted information withholding techniques, providing a rich training playground that could be advantageous in some settings, notably data-scarce ones. I then address subtle issues that may arise when porting reinforcement learning techniques to the diffusion context, and wonder how such exploration problems could be addressed in more diffusion-native ways. I do not have definitive answers, but I do point my fingers in directions I deem interesting. A tutorial follows this thesis, expanding on the destroy-then-generate perspective. A novel kind of probabilistic graphical models is introduced to facilitate the tutorial's exposition.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
    enabled: true
    zoomable: false

# Anonymize when submitting
authors:
    - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-27-destruction.bib

toc:
    - name: The Thesis
      subsections:
        - name: Learning by Destroying
        - name: Diffusion's Advantages
        - name: Reconciling Exploration
    - name: The Tutorial
      subsections:
        - name: Destroying and Generating
        - name: Mashing Everything
        - name: Shuffling Everything
        - name: Beyond Markov
        - name: About the Diagrams

# Style injection to make mermaid flowcharts readable during redaction.
# Some more elegant solution should be used in an eventual accepted version.
_styles: >
  .mermaid svg {
      max-width: 1200px !important;
  }
---

> Pourquoi faire simple quand on peut faire compliqué?
> 
> --French locution, usually ironic.


This blogpost is composed of two main sections, tied together by their unusual information-theoretic viewpoint.
[The Thesis](#the-thesis) is an opinion/perspective/speculative piece on diffusion models.
[The Tutorial](#the-tutorial) is a diagrammatic presentation of diffusion models.


# The Thesis

## Learning by Destroying

Yang *et al.*<d-cite key="yang2023diffusion"></d-cite> defined generative diffusion models as "a family of probabilistic generative models that progressively destruct data by injecting noise, then learn to reverse this process for sample generation."
I think that it is a good definition, except for the "by injecting noise" part.
Indeed, there is a paper that is literally called *Cold Diffusion: Inverting Arbitrary Image Transforms **Without Noise***<d-cite key="bansal2023cold"></d-cite> (emphasis mine), and calling "noise" the action of the `[MASK]` token in Mask Diffusion Language Models (MDLMs)<d-cite key="sahoo2024simple"></d-cite> is certainly debatable.
For the sake of this blogpost, I will define generative diffusion models as follows.

1. **Data distribution.** We have a training dataset sampled from a data distribution, and our goal is to generate new samples from the same data distribution. The only allowed "control mechanism" is *conditioning*.<d-footnote>Concretely, the condition can be specified as a predicate: a function accepting a data sample and returning either "true" or "false". When such a condition is provided, the model must only return samples for which the predicate returns "true", otherwise keeping the data distribution unaltered. Notice that this could be achieved by wrapping the model in a loop, and returning the first samples that passes the predicate check.</d-footnote>

2. **Destroying process.** We also have access to a procedure that can gradually destroy the information in data samples. This procedure **may** involve randomness. Like the training dataset, this procedure if fully specified *a priori*, before any training takes place.

3. **Generative process.** We train a generative process so that the samples it generates approximate the sought data distribution. This training leverages both the training dataset, which specifies *what* the generative process should ultimately generate, and the destroying process, which specifies *how* to chunk generation into manageable pieces.<d-footnote>The loss function being minimized is typically an expectation of an expectation of an expectation, with those expectations respectively taken over the training data sample, the "level of destruction" to be applied on this training data sample, and the exact stochastic realization of this destruction (if applicable).</d-footnote>

Using this definition, diffusion models can be seen as a special case of the tried-and-true machine learning technique consisting of withholding some information from a model's input, then training the model to generate the withheld information.
- In supervised learning, the withheld (then generated) information is called "label".
- In an autoregressive language model, it is the next token.
- For BERT-like language models, it is the `[MASK]`ed tokens.
- Some autoencoders rely on a representational bottleneck to withhold information.
- Contrastive autoencoders withhold whether pairs of samples are somehow related (positive pairs) or not (negative pairs).
- Denoising autoencoders add "noise" to samples, which destroys some of their informational content (i.e., withholding it).
- Diffusion models use the destroying process to withhold information.<d-footnote>This typically involves some form of randomness, but here we do not make it a requirement.</d-footnote>

You may notice a connection between the last two: that connection has already been made long ago.<d-cite key="kingma2021variational"></d-cite> 
In fact, such connections have been made between diffusion and other points in that list, <d-cite key="zheng2025masked,fathi2025unifying"></d-cite> and I think that, given enough coffee, we could get them all covered.
*Should we*?
That's a good question, and the answer may depend in part on how data-starved we are.

## Diffusion's Advantages

In *Diffusion Language Models are Super Data Learners*, Ni *et al.*<d-cite key="ni2025diffusion"></d-cite> observe that

> when unique data is limited, diffusion language models (DLMs) consistently surpass autoregressive (AR) models by training for more epochs. The crossover shifts later with more or higher-quality data, earlier with larger models, and persists across dense and sparse architectures. We attribute the gains to three compounding factors: (1) any-order modeling, (2) super-dense compute from iterative bidirectional denoising, and (3) built-in Monte Carlo augmentation [...]

I think that this makes a lot of sense, and that **this phenomenon is likely general**.
We have devised simple (non-diffusion) ways to withhold information from our models, breaking the data into chunks that "make sense" to us, emphasizing aspects that must obviously be learned sooner or later.
And when we train models to generate that withheld information, these simple approaches turn out to be locally optimal for the sake of training as quickly as possible on humongous datasets.
But when the fresh data gets scarce, when you have to train from the same sample for the $N^{\text{th}}$ time, perhaps more could be learned by considering a different viewpoint -- like a diffusion model learning to predict tokens out-of-order.
To be clear, ultimately, it could be that the diffusion model will perform better during inference when it is used to predict tokens in order, like an autoregressive model would simply do.<d-cite key="kim2025train"></d-cite>
But because it was *challenged* during training on out-of-order tasks, it may eventually manage to pick up some tricks that will forever evade the autoregressively-trained model.
Bigger models have more capacity to latch on such tricks, so their crossover may come earlier.

Depending on your perspective, this may align with Sutton's *The Bitter Lesson*:<d-cite key="sutton2019bitter"></d-cite>

> general methods that leverage computation are ultimately the most effective, and by a large margin. [...] Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation.

Left-to-right training may have been yet another human bias, a temporary fad, and in the long run perhaps we'll just train in all-the-orders, leveraging much more compute.
But there is a sweeter perspective on that bitter lesson: the design space for novel diffusion models is huge.
To this day, very few destroying processes have been considered.
For example, `[MASK]` is a great candidate for the next human fad/bias to be replaced by something that better leverages compute.
Moreover, while diffusion may beat our vanilla information-withholding approaches in the bounded data regime, it hasn't been shown to be the best way to proceed: diffusion itself may be a weird human fad.

## Reconciling Exploration

This takes me to the last point of this blogpost's thesis: if all incarnations of the tried-and-true "information withholding" machine learning technique can (and perhaps should) be related to diffusion, what other techniques are left?
And could we improve upon diffusion by learning from them?
The first answers that come to my mind are "anything that involves exploration," and "yes, probably."

The archetypal technique involving exploration is Reinforcement Learning (RL).
Whereas pure generative models solely learn from a training dataset -- striving to generate new samples from the exact same data distribution -- RL models strive for a different, "better" distribution.
What is meant by "better" is usually (but not always) specified through a reward function: a function that assigns value (reward) to each possible data sample.
We typically wish to maximize the expected reward, though there may be some requirement to not meander too far away from the original data distribution.

GRPO<d-cite key="shao2024deepseekmath"></d-cite> is a quite successful RL technique, and it has recently been ported to diffusion.<d-cite key="gong2025diffucoder"></d-cite>
Such an adaptation is not trivial: due to its autoregressive origin, GRPO takes for granted easy access to likelihoods for different sequences, but such likelihoods are harder to get in diffusion models.
This issue has been overcome before in the context of perplexity<d-cite key="sahoo2024simple"></d-cite>: take expectations over different levels of destruction and over different destruction realization, which translates to different decoding orders.
Gong *et al.*<d-cite key="gong2025diffucoder"></d-cite> leverage this technique,<d-footnote>With the twist that when a decoding order is used in the estimation of this expectation, the "complement" of the mask associated to that decoding order is always also considered.
This practice reduces the variance, allowing them to take expectations over a single such pair of orders.</d-footnote>
and their results are great!

Yet I suspect that this expectation-over-decoding-order strategy is inherently off-policy.
Indeed, even for purely random decoding orders,<d-footnote>To be clear, I definitely believe that it would be a good idea to learn a policy for the decoding order. I'm here assuming a non-learnable random order to make my point stronger.</d-footnote> I claim that the specific order faced by the model while generating a trace should be accounted for in the reward assignment for that trace.
My intuition goes as follows: if a successful reasoning trace summarizes some premises from the context, expands some methodic steps, then reaches some conclusion, what are we teaching the model by rewarding it to predict the conclusion first, without the reasoning steps that lead to it?<d-footnote>My answer: at best we're teaching it to skip steps and/or ignore reasoning, at worst we're teaching it to hallucinate, make stuff up, justify *a posteriori* and/or otherwise deceive.</d-footnote>

You may now think "Who are you to say how the model should or shouldn't reason! Remember The Bitter Lesson!"
Fair enough, but here's my point: in the non-RL diffusion case, this expectation-over-decoding-order approach was grounded in sound theory, but we didn't do our homework before porting it to RL.
We can justify it by its good empirical results, but we lost our theoretical grounding.
*A priori*, the only path we may reward on-policy for a given generated sample is the one that was followed by the model while generating that sample.

Ok, how did we get that theoretical grounding in the non-RL case?
Limiting ourselves to my strict definition at the beginning of this blogpost, we consider a specific data distribution and noising process, and there thus exists a single, ideal, typically untractable<d-footnote>This is why we have to train a neural network: we're learning a tractable function that approximates the untractable ideal one.</d-footnote> probability distribution for partially-destroyed data samples at different levels of destruction.
Within a given modeling paradigm, all concrete diffusion model implementations seek to approximate some function of that same ideal distribution: models with different weight initialization (or even different neural network architectures!) all strive to approximate the same "correct" answer.
This is a very convenient property: in a diffusion model (as per my definition), the function to be learned is not a moving target.

Can we approach the RL problem with a diffusion model that satisfies this strict definition?
Yes!
By reframing it as conditioning,<d-cite key="yuan2023reward"></d-cite> which is the sole allowed control mechanism as per my definition.
For now, work on that front is still in an early stage, and competition leveraging techniques ported from the autoregressive case (e.g., GRPO) have an head start.
Moreover, there is no guarantee that approaching RL by sticking to my strict definition of diffusion has real advantages in the long run.
Nonetheless, I think that the fact we *can* suffices to justify additional exploratory efforts.

More generally, notice how reframing as a conditioning problem removed "exploration" from the picture: we're not *searching* for high-reward samples, we're just *filtering out* from the original data distribution the samples that have low reward.
In practice, because we don't have infinite resources, actual implementations still have to explore to find where it is worth it to learn the function.
We have the guaranteed existence of a non-moving target function, but we have to find which parts of that target function are worth learning well.<d-footnote>Recall that "standard" RL models are often constrained to not meander too far away from the original data distribution. Concretely, this is usually implemented by comparing the predictions of a frozen, "old" model with those of its RLed counterpart. In the conditioned diffusion approach to RL, the non-moving target function plays the role of that old model.</d-footnote>

The astute readers may notice that we've been infringing on a second taboo of my strict definition: the destroying process is *given* (not learned!) before any training takes place, and it specifies what/how the model should learn.
Again, there are workarounds: we may tweak how we sample the destruction process while maintaining the same nice, non-moving target.<d-footnote>For example, we may counterbalance frequency biases by inversely weighting the losses.</d-footnote>
But is there a point where we could gain something by giving up that nice, non-moving target?

Of course!
We move that target every time a researcher comes up with a "better" destroying process, and there is no good reason to believe that this human-in-the-loop algorithm has already found the optimum.
The data should speak for itself!
The compute should be leveraged!
Umh, how?

Let me greatly simplify the problem.

1. **First distribution.** We have a data distribution from which we can get a training dataset.

2. **Second distribution.** We have a destroying process whose specifics depends on parameters to be determined. However, whatever these parameters are, we already know what the probability distribution for a "fully destroyed" data distribution should look like.<d-footnote>For example, a Gaussian.</d-footnote>

3. **Cost function.** For given parameters of the destroying process, we train a diffusion model to generate from the data distribution. We are given a function that associates a cost to these parameters: the cost may depend on the ultimate performances of the model when transporting from the fully-destroyed distribution to the data distribution, but also on the model's size and the resources it consumed at training and/or inference. We seek to minimize the expectation of this cost.

This is an instance of Optimal Transport (OT) problems, which are traditionally introduced with piles of dirt.
Imagine a pile of dirt whose height profile represents a probability distribution (higher probability means more dirt piled up there).
We want to move the dirt around so that it represents a second distribution instead.
There are many ways to do so, many exact plans for where to pick each shovelful of dirt and where to toss them.
We can assign a cost<d-footnote>That is, a negative reward.</d-footnote> to each such plan, and we must explore the space of plans to find the cheapest.<d-footnote>If you have ever found yourself sitting in a conference room while the speaker said "... Wasserstein metric, also known as earth mover's distance...", then the talk was likely about OT.</d-footnote>

Ok, but is there anything practical to gain by framing the search for the best destroying process in terms of an OT problem?
Well, I don't know.
But perhaps I should now mention that entropy-regularized OT is tightly related to diffusion,<d-cite key="debortoli2021diffusion"></d-cite> a fact that definitely weights in my intuition that there may be something interesting to be done here.
Again, I think that the fact we *can* suffices to justify some exploratory efforts.

So destruction is a general strategy for learning to generate information about a data distribution, diffusion's ~~messiness~~ richness may advantageously leverage destruction in some contexts (including data-starved ones), exploring how to optimize a function doesn't neatly fit this pictures, but there are workarounds reconciliating diffusion and exploration, opening a multitude of avenues for future work.
There are no promises, but we should give it a try.


# The Tutorial

## Destroying and Generating

This section provides a diagrammatic introduction to "destroying" and "generating" information.

As is tradition in information theory, we consider two characters, Alice and Bob.
Alice communicates one of 3 messages to Bob through a communication channel.
A big arrow indicates the name of the channel itself, `Identity`, while smaller arrows indicate the action of that channel.

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
        b2[2]
    end
    a0 --> b0
    a1 --> b1
    a2 --> b2

    A ==Identity==> B

    LeftMarginHack:::phantom ~~~ a0
    b0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

In the above situation, Bob gets exactly the message that Alice tried to convey.
The information communicated by Alice is thus preserved (i.e., neither created nor destroyed) by the `Identity` communication channel: it is an important (albeit boring) channel.

Ok, now let's try a different channel, which I'll call `Cypher`.

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b1[1]
        b2[2]
        b0[0]
    end
    a0 --> b1
    a1 --> b2
    a2 --> b0

    A ==Cypher==> B

    LeftMarginHack:::phantom ~~~ a0
    b0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Here Bob gets a different number than the one Alice intended to send.
Does this mean that `Cypher` destroys information?
Well, it would if the channel could only be used once before being discarded...

But for the sake of this blogpost, let's instead consider the case where Bob is allowed as many training examples as he needs.
During that training, Bob gets Alice's message through both the trusty `Identity` channel and the to-be-figured-out `Cypher` channel.
Using pairs of the form $(a,b)$ (with $a \in A$ and $b \in B$), Bob's observations may look like

$$
\{ (0,1), (2,0), (0,1), (0,1), (1,2), (0,1), (2,0), (0,1), (2,0), (0,1), (0,1), (0,1), \cdots \}.
$$

There are many kinds of things that Bob could learn from such data.
First, he may hypothesize that Alice is more likely to give the message $0$ than she is to say $1$ or $2$.
In doing so, Bob would be building a mental model of an Imaginary-Alice $A'$, i.e., trying to learn the probability distribution $\textup{P}^\theta(A')$ so that it matches the real $\textup{P}(A)$.
And ultimately, this is exactly what generative modeling is about: learn how to sample from a $\textup{P}^\theta(A')$ that approximates as best as we can the data distribution $\textup{P}(A)$.

Doing so may be realistic for $3$ messages, but what if Alice had more range, say, $10^{678000}$ possible unique messages?<d-footnote>GPT OSS 120B's vocabulary size powered to its context length.</d-footnote>
This direct approach won't scale, which is why the next section will consider a divide-and-conquer approach to chunk large problems into more amenable ones.
For now, it suffices to say that Bob is just a step in a long chain of messages, followed by Carol, David, *etc.*
If each step is easier to model/learn than the previous one, we'll be making progress toward our ultimate goal.

So, Bob should focus on figuring out the `Cypher` channel, i.e., learn a probability distribution $\textup{P}^{\theta}(A \vert B)$
that approximates $\textup{P}(A \vert B)$.

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b1[1]
        b2[2]
        b0[0]
    end
    subgraph Ap["A'"]
        ap0[0]
        ap1[1]
        ap2[2]
    end
    a0 --> b1 --> ap0
    a1 --> b2 --> ap1
    a2 --> b0 --> ap2

    A ==Cypher==> B
    B ==Decrypt==> Ap
    A ==Identity==> Ap

    LeftMarginHack:::phantom ~~~ a0
    ap0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

There exists a `Decrypt` channel that, applied on the output of the `Cypher`, gives the `Identity`.
If Bob is a good Bayesian, he will never be *absolutely sure* that he figured it out, but whatever his priors were<d-footnote>Well, within reason, assuming that Bob has a minimal pragmatism and/or understanding of the world...</d-footnote> for the probability distribution from which the `Cypher` channel was sampled, there is a point at which `Decrypt` will become his leading hypothesis as to what $\textup{P}^\theta(A' \vert B)$ should be.
And from that point on, his confidence in that hypothesis will keep increasing as more data is gathered.

The existence of a `Decrypt` that reverses the action of `Cypher` proves that `Cypher` does not destroy information.
And because we could do the opposite, i.e., reverse the action of `Decrypt` by applying `Cypher`, we know that `Cypher` does not generate information either.
Just like `Identity`, `Cypher` preserves information.
What does this mean?
It means that `Cypher` is a useless channel for our divide-and-conquer aims: learning a $\textup{P}^\theta(B')$ that approaches $\textup{P}(B)$ is as hard as learning a $\textup{P}^\theta(A')$ that approaches $\textup{P}^\theta(A)$.

Ok, maybe what we need is a noisy channel?

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
        b2[2]
        b3[3]
        b4[4]
        b5[5]
    end
    a0 --head--> b0
    a0 --tail--> b3
    a1 --head--> b1
    a1 --tail--> b4
    a2 --head--> b2
    a2 --tail--> b5

    A ==Product==> B

    LeftMarginHack:::phantom ~~~ a0
    b0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Here the `Product`<d-footnote>This name is a reference to category theory; please see the very last section for why.</d-footnote> channel flips a coin, and this affects what message Bob receives.
Does `Product` destroy information?

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
        b2[2]
        b3[3]
        b4[4]
        b5[5]
    end
    subgraph Ap["A'"]
        ap0[0]
        ap1[1]
        ap2[2]
    end
    a0 --head--> b0 --> ap0
    a0 --tail--> b3 --> ap0
    a1 --head--> b1 --> ap1
    a1 --tail--> b4 --> ap1
    a2 --head--> b2 --> ap2
    a2 --tail--> b5 --> ap2

    A ==Product==> B ==Project==> Ap
    A ==Identity==> Ap

    LeftMarginHack:::phantom ~~~ a0
    ap0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

No: there exists a `Project` channel that undoes the action of `Product`.
Does it generate information?

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b3[3]
        b1[1]
        b4[4]
        b2[2]
        b5[5]
    end
    subgraph Coin
        head
        tail
    end
    a0 --head--> b0 --> head
    a0 --tail--> b3 --> tail
    a1 --head--> b1 --> head
    a1 --tail--> b4 --> tail
    a2 --head--> b2 --> head
    a2 --tail--> b5 --> tail

    A ==Product==> B ==Project'==> Coin

    LeftMarginHack:::phantom ~~~ a0
    head ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Yes! Bob could use `Project'` to learn the outcome of the coin flip, an information that Alice is completely unaware of!

For our divide-and-conquer aims, we needed Bob to be easier to model than Alice, and we got the opposite: Bob is strictly harder to model than Alice because he has all of Alice's information, plus some irrelevant information about a coin.
Therefore, `Product` is an even worse channel than `Cypher` for this divide-and-conquer purpose.

If noise isn't what we need, what is it?

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
    end
    a0 --> b0
    a1 --> b1
    a2 --> b1

    A ==Mash==> B

    LeftMarginHack:::phantom ~~~ a0
    b0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

That's it!
When Bob gets the message $1$, he can't tell with certainty whether Alice said $1$ or $2$.
`Mash`ing different messages together is what destroys information, not noise.

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
    end
    subgraph Ap["A'"]
        ap0[0]
        ap1[1]
        ap2[2]
    end
    a0 --> b0 --> ap0
    a2 --> b1 --> ap1
    a1 --> b1 --> ap2

    A ==Mash==> B ==Guess==> Ap
    A -.B.- Ap

    LeftMarginHack:::phantom ~~~ a0
    ap0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

When Bob gets the message $B=0$, he can simply assign $A'=0$.
But when $B=1$, he must learn to `Guess` the message sent by an Imaginary-Alice $A'$ so that $\textup{P}^\theta(A'|B)$ is as close as possible to $\textup{P}(A|B)$.

I indicate this last requirement using a dashed line, with the conditional $B$ added on the line.
However, in many applications, we don't have a particular desire for $\textup{P}^\theta(A'|B) = \textup{P}(A|B)$, and we could be perfectly content with any $\textup{P}^\theta(A'|B)$ such that $\textup{P}^\theta(A') = \textup{P}(A)$.
In those cases, I simply don't write anything on the dashed line.

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
    end
    subgraph Ap["A'"]
        ap0[0]
        ap1[1]
        ap2[2]
    end
    a0 --> b0 --> ap0
    a2 --> b1 --> ap0
    a1 --> b1
    b0 --> ap1
    b1 --> ap1
    b0 --> ap2
    b1 --> ap2

    A -.- Ap
    A ==Mash==> B ==> Ap
    

    LeftMarginHack:::phantom ~~~ a0
    ap0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Note that I didn't name the arrow from Bob to the Imaginary-Alice, because I don't really care how Bob does it: as long as $\textup{P}^\theta(A') = \textup{P}(A)$, i.e., the two people joined by an un-decorated dashed line have the same marginal distribution, we're good.

And I may not even care about the details of the possible messages and how they relate (small arrows).

```mermaid
flowchart LR
    subgraph A
        a0:::phantom
    end
    subgraph B
        b0:::phantom
    end
    subgraph Ap["A'"]
        ap0:::phantom
    end
    A ==> B ==> Ap
    A -.- Ap

    LeftMarginHack:::phantom ~~~ A
    Ap ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

What does this picture tell us?
Well, for a starter, because $A$ and $A'$ have the same marginal probability distribution, they must have the same information content (entropy)

$$
\textup{H}(A) = \textup{H}(A') .
$$

How does $\textup{H}(B)$ relate to those two?
We don't know: it could be higher, lower, or equal, depending on the details of those communication channels.
One thing is clear though: as much information must be created/destroyed from $A$ to $B$ as is destroyed/created from $B$ to $A'$.

This picture can tell us much more than that, because of the data processing inequality.
From Latham and Roudi<d-cite key="Latham2009"></d-cite>:

> The Data Processing Inequality (DPI) states, loosely, that post-processing cannot increase information.

If you think "Haven't we just shown that communication channels may create information?!", then yes, you are correct.
But the information created by communication channels *is completely irrelevant to anything that came before in the communication chain*.
Latham and Roudi are saying that information about an early message cannot appear in later messages by further processing.
This "information about" can be measured with the mutual information; we write $\textup{I}(Y; X)$ the mutual information between $X$ and $Y$, and it satisfies

$$
\textup{I}(Y; X) = \textup{H}(Y) - \textup{H}(X | Y) = \textup{H}(X) - \textup{H}(Y | X)  , \qquad \textup{I}(Y; Y) = \textup{H}(Y) .
$$

```mermaid
flowchart LR
    subgraph A
        a0:::phantom
    end
    subgraph B
        b0:::phantom
    end
    subgraph Ap["A'"]
        ap0:::phantom
    end
    A ==> B ==> Ap

    LeftMarginHack:::phantom ~~~ A
    Ap ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Therefore, what the data processing inequality tells us is that

$$\eqalign{
\textup{I}(A; A) \ge \textup{I}(A; B) & \ge \textup{I}(A; A') \cr
\textup{I}(B; B) & \ge \textup{I}(B; A') .
}$$

Whatever there is to know about $A$, $B$ cannot know more than that, and $A'$ cannot know more than $B$ did.
And $A'$ cannot know more about $B$ than $B$ knew about himself.

This is what I call "the destruction story": **you cannot know what has been forgotten before you heard about it.**
However, the data processing inequality has a lesser known dual, which I call "the generation story"

$$\eqalign{
\textup{I}(A'; A) & \le \textup{I}(A'; B) \le \textup{I}(A'; A') \cr
\textup{I}(B; A) & \le \textup{I}(B; B) .
}$$

$A$ cannot know more than $B$ about $A'$, and $B$ cannot know more about $A'$ than there is to be known about $A'$.
And $A$ cannot know more than $B$ about $B$.
In the generation story, **you cannot know about what has not been decided yet.**

These zoomed-out diagrams are a great way to consider general classes of problems.
But ultimately, the generation/destruction of information happens at the zoomed-in, message level.

```mermaid
flowchart LR
    focus[" "]
    
    leftt[" "] --> focus --> rightt[" "]
    leftc:::phantom ~~~|"destroy"| focus ~~~|"generate"| rightc:::phantom
    leftb[" "] --> focus --> rightb[" "]

    LeftMarginHack:::phantom ~~~ leftt
    rightb ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

- When different messages converge to the same message, information is **destroyed**.
- When a given message can diverge into different messages, information is **generated**.


## Mashing Everything

Let's continue from the `Mash` example in the previous section.
Now suppose that Bob passes the message to Carol through the `Mash'` channel.

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
    end
    subgraph C
        c0[0]
    end
    subgraph Singleton[*]
        singleton[" "]
    end
   
    a0 --> b0 --> c0
    a1 --> b1 --> c0
    a2 --> b1

    A ==Mash==> B ==Mash'==> C -.- Singleton

    LeftMarginHack:::phantom ~~~ a0
    singleton ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Notice that Carol always gets the same message, which is called a singleton: I highlight this by linking her to a special "*" symbol using a dashed line.
All singleton have no information: $\textup{H}(C) = 0$.

But by the definition of mutual information, we must have $\textup{I}(A; C) \le \textup{H}(C) = 0$: Carol cannot know more about Alice than Carol knows at all, but Carol knows nothing, therefore Carol knows nothing about Alice.
Stated otherwise, all information about Alice has been destroyed by the time the messages get to Carol.
Success!
We have divided a large task into smaller, simpler ones, until all that was left was trivial!

Now unto the "conquer" part of divide-and-conquer.

```mermaid
flowchart LR
    subgraph A
        a0:::phantom
    end
    subgraph B
        b0:::phantom
    end
    subgraph Ap["A'"]
        ap0:::phantom
    end
    subgraph C
        c0:::phantom
    end
    subgraph Bp["B'"]
        bp0:::phantom
    end
    subgraph Singleton["*"]
        singleton:::phantom
    end
    A ==Mash==> B ==Mash'==> C ==Guess'==> Bp
    B ==Guess==> Ap
    A -.- Ap
    B -.- Bp
    C -.- Singleton

    LeftMarginHack:::phantom ~~~ a0
    bp0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Bob can learn `Guess` using `Mash` on Alice's message, and Carol can learn `Guess'` using `Mash'` on Bob's.
And because Carol knows nothing, we're ready to generate completely fake $A''$ behavior from nothing! 

```mermaid
flowchart LR
    subgraph Singleton["*"]
        singleton:::phantom
    end
    subgraph Bpp["B''"]
        bpp0:::phantom
    end
    subgraph App["A''"]
        app0:::phantom
    end
    subgraph A["A"]
        a0:::phantom
    end
    
    Singleton =="Guess'"==> Bpp ==Guess==> App -.- A

    LeftMarginHack:::phantom ~~~ Singleton
    a0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

This is the strategy used in autoregressive language models.

```mermaid
flowchart LR
    subgraph A
        a0["('The', 'cat', 'is', 'black')"]
        a1["('The', 'cat', 'is', 'white')"]
        a2["('The', 'cat', 'is', 'sleepy')"]
        a3["('The', 'cat', 'was', 'sleepy')"]
        a4["('The', 'dog', 'is', 'black')"]
        a5["('The', 'dog', 'is', 'brown')"]
        a6["('Your', 'cat', 'is', 'black')"]
        a7["..."]
    end
    subgraph B
        b0["('The', 'cat', 'is')"]
        b1["('The', 'cat', 'was')"]
        b2["('The', 'dog', 'is')"]
        b3["('Your', 'cat', 'is')"]
        b4["..."]
    end
    subgraph C
        c0["('The', 'cat')"]
        c1["('The', 'dog')"]
        c2["('Your', 'cat')"]
        c3["..."]
    end
    subgraph D
        d0["('The')"]
        d1["('Your')"]
        d2["..."]
    end
    subgraph E
        e0["()"]
    end
   
    a0 --> b0 --> c0 --> d0 --> e0
    a1 --> b0
    a2 --> b0
    a3 --> b1 --> c0
    a4 --> b2 --> c1 --> d0
    a5 --> b2
    a6 --> b3 --> c2 --> d1 --> e0

    LeftMarginHack:::phantom ~~~ a0
    e0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Here the ellipses represent additional messages whose arrows are not explicitly shown.
The message is iteratively fed through the `MashLastToken` communication channel, which mashes together all the messages that have the same prefix up to the last token.
Because of the setting's symmetric structure, we can use the same neural network at each position to predict the probability distribution for what the next should be: all positions can contribute to train the same `GuessNextToken`.
At inference, we start from the singleton (the empty sequence), then iteratively apply `GuessNextToken`.
Et voilà!
An autoregressive language model!

Notice how each time we destroy some information, we're carving out a chunk of the overall task to be learned.
Here divide-and-conquer amounted to "learn to generate one token at a time".
Such a very structured, regular way to destroy the message, is typical of many non-diffusion machine learning techniques.
One of my claims in this blogpost is that diffusion models can destroy information more "organically", possibly reducing the human-designer bias hand-waved at by Sutton.<d-cite key="sutton2019bitter"></d-cite>
As an example of what I mean, here's what the last diagram would look like for a mask diffusion language model.

```mermaid
flowchart LR
    subgraph A
        a0["('The', 'cat', 'is', 'black')"]
        a1["('The', 'cat', 'is', 'white')"]
        a2["('The', 'cat', 'is', 'sleepy')"]
        a3["('The', 'cat', 'was', 'sleepy')"]
        a4["('The', 'dog', 'is', 'black')"]
        a5["('The', 'dog', 'is', 'brown')"]
        a6["('Your', 'cat', 'is', 'black')"]
        a7["..."]
    end
    subgraph B
        b03["('The', 'cat', 'is', '[MASK]')"]
        b33["('The', 'cat', 'was', '[MASK]')"]
        b43["('The', 'dog', 'is', '[MASK]')"]
        b63["('Your', 'cat', 'is', '[MASK]')"]
        
        b02["('The', 'cat', '[MASK]', 'black')"]
        b12["('The', 'cat', '[MASK]', 'white')"]
        b22["('The', 'cat', '[MASK]', 'sleepy')"]
        b42["('The', 'dog', '[MASK]', 'black')"]
        b52["('The', 'dog', '[MASK]', 'brown')"]
        b62["('Your', 'cat', '[MASK]', 'black')"]
        
        b01["('The', '[MASK]', 'is', 'black')"]
        b11["('The', '[MASK]', 'is', 'white')"]
        b21["('The', '[MASK]', 'is', 'sleepy')"]
        b31["('The', '[MASK]', 'was', 'sleepy')"]
        b51["('The', '[MASK]', 'is', 'brown')"]
        b61["('Your', '[MASK]', 'is', 'black')"]
        
        b00["('[MASK]', 'cat', 'is', 'black')"]
        b10["('[MASK]', 'cat', 'is', 'white')"]
        b20["('[MASK]', 'cat', 'is', 'sleepy')"]
        b30["('[MASK]', 'cat', 'was', 'sleepy')"]
        b40["('[MASK]', 'dog', 'is', 'black')"]
        b50["('[MASK]', 'dog', 'is', 'brown')"]
        
        b7["..."]
    end

    a0 --> b03
    a1 --> b03
    a2 --> b03
    a3 --> b33 
    a4 --> b43 
    a5 --> b43
    a6 --> b63

    a0 --> b02
    a1 --> b12
    a2 --> b22
    a3 --> b22
    a4 --> b42
    a5 --> b52
    a6 --> b62

    a0 --> b01
    a1 --> b11
    a2 --> b21
    a3 --> b31
    a4 --> b01
    a5 --> b51
    a6 --> b61

    a0 --> b00
    a1 --> b10
    a2 --> b20
    a3 --> b30
    a4 --> b40
    a5 --> b50
    a6 --> b00

    subgraph C
        c023["('The', 'cat', '[MASK]', '[MASK]')"]
        c423["('The', 'dog', '[MASK]', '[MASK]')"]
        c623["('Your', 'cat', '[MASK]', '[MASK]')"]

        c013["('The', '[MASK]', 'is', '[MASK]')"]
        c313["('The', '[MASK]', 'was', '[MASK]')"]
        c613["('Your', '[MASK]', 'is', '[MASK]')"]

        c003["('[MASK]', 'cat', 'is', '[MASK]')"]
        c303["('[MASK]', 'cat', 'was', '[MASK]')"]
        c403["('[MASK]', 'dog', 'is', '[MASK]')"]
        
        c023["('The', 'cat', '[MASK]', '[MASK]')"]
        c423["('The', 'dog', '[MASK]', '[MASK]')"]
        c623["('Your', 'cat', '[MASK]', '[MASK]')"]

        c012["('The', '[MASK]', '[MASK]', 'black')"]
        c112["('The', '[MASK]', '[MASK]', 'white')"]
        c212["('The', '[MASK]', '[MASK]', 'sleepy')"]
        c512["('The', '[MASK]', '[MASK]', 'brown')"]
        c612["('Your', '[MASK]', '[MASK]', 'black')"]

        c002["('[MASK]', 'cat', '[MASK]', 'black')"]
        c102["('[MASK]', 'cat', '[MASK]', 'white')"]
        c202["('[MASK]', 'cat', '[MASK]', 'sleepy')"]
        c402["('[MASK]', 'dog', '[MASK]', 'black')"]
        c502["('[MASK]', 'dog', '[MASK]', 'brown')"]
        
        c001["('[MASK]', '[MASK]', 'is', 'black')"]
        c101["('[MASK]', '[MASK]', 'is', 'white')"]
        c201["('[MASK]', '[MASK]', 'is', 'sleepy')"]
        c301["('[MASK]', '[MASK]', 'was', 'sleepy')"]
        c501["('[MASK]', '[MASK]', 'is', 'brown')"]
        
        c7["..."]
    end
    B ==> C
    subgraph D
        d0012["('[MASK]', '[MASK]', '[MASK]', 'black')"]
        d1012["('[MASK]', '[MASK]', '[MASK]', 'white')"]
        d2012["('[MASK]', '[MASK]', '[MASK]', 'sleepy')"]
        d5012["('[MASK]', '[MASK]', '[MASK]', 'brown')"]

        d0013["('[MASK]', '[MASK]', 'is', '[MASK]')"]
        d3013["('[MASK]', '[MASK]', 'was', '[MASK]')"]

        d0023["('[MASK]', 'cat', '[MASK]', '[MASK]')"]
        d4023["('[MASK]', 'dog', '[MASK]', '[MASK]')"]

        d0123["('The', '[MASK]', '[MASK]', '[MASK]')"]
        d6123["('Your', '[MASK]', '[MASK]', '[MASK]')"]

        d7["..."]
    end
    c023 --> d0023
    c423 --> d0023
    c013 --> d0013
    c313 --> d3013
    c613 --> d0013
    c003 --> d0013
    c303 --> d3013
    c403 --> d0013
    c623 --> d0023
    c012 --> d0012
    c112 --> d1012
    c212 --> d2012
    c512 --> d5012
    c612 --> d0012
    c002 --> d0012
    c102 --> d1012
    c202 --> d2012
    c402 --> d0012
    c502 --> d5012
    c001 --> d0012
    c101 --> d1012
    c201 --> d2012
    c301 --> d2012
    c501 --> d5012

    c023 --> d0123
    c423 --> d0123
    c013 --> d0123
    c313 --> d0123
    c613 --> d6123
    c003 --> d0023
    c303 --> d0023
    c403 --> d4023
    c623 --> d6123
    c012 --> d0123
    c112 --> d0123
    c212 --> d0123
    c512 --> d0123
    c612 --> d6123
    c002 --> d0023
    c102 --> d0023
    c202 --> d0023
    c402 --> d0023
    c502 --> d0023
    c001 --> d0013
    c101 --> d0013
    c201 --> d0013
    c301 --> d3013
    c501 --> d0013

    subgraph E
        e0["('[MASK]', '[MASK]', '[MASK]', '[MASK]')"]
    end
    d0012 --> e0
    d1012 --> e0
    d2012 --> e0
    d5012 --> e0

    d0013 --> e0
    d3013 --> e0

    d0023 --> e0
    d4023 --> e0

    d0123 --> e0
    d6123 --> e0
   
    LeftMarginHack:::phantom ~~~ a0
    e0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

(There should be 3 arrows leaving each of Bob's messages toward Carol, please pardon me for eschewing them.)

While `MashLastToken` was strictly destroying information, here `MashOneRandomUnmaskedToken` both generates information (which token is to be masked) and destroys information (mash together all messages that would be the same if it were not for the randomly-selected unmasked token).
However, the newly generated information (masking order) is also destroyed by the process.
In the end, we still get a singleton: all information is eventually destroyed.

The point I'm trying to make here is that `MashOneRandomUnmaskedToken` is a richer approach to "divide and conquer" than `MashLastToken` was.
It is more "organic" in a similar sense that mixing milk in coffee is "organic": it's a mess, a *rich* mess.
We can train a model to `GuessOneRandomMaskToken` on that mess, and yes, I could excuse it to train slower than `GuessNextToken`, because `GuessNextToken` is a subset of what `GuessOneRandomMaskToken` is learning.
Let me make it explicit.

```mermaid
flowchart LR
    subgraph A
        a0["('The', 'cat', 'is', 'black')"]
        a1["('The', 'cat', 'is', 'white')"]
        a2["('The', 'cat', 'is', 'sleepy')"]
        a3["('The', 'cat', 'was', 'sleepy')"]
        a4["('The', 'dog', 'is', 'black')"]
        a5["('The', 'dog', 'is', 'brown')"]
        a6["('Your', 'cat', 'is', 'black')"]
        a7["..."]
    end
    subgraph B
        b0["('The', 'cat', 'is', '[MASK]')"]
        b1["('The', 'cat', 'was', '[MASK]')"]
        b2["('The', 'dog', 'is', '[MASK]')"]
        b3["('Your', 'cat', 'is', '[MASK]')"]
        b4["..."]
    end
    subgraph C
        c0["('The', 'cat', '[MASK]', '[MASK]')"]
        c1["('The', 'dog', '[MASK]', '[MASK]')"]
        c2["('Your', 'cat', '[MASK]', '[MASK]')"]
        c3["..."]
    end
    subgraph D
        d0["('The', '[MASK]', '[MASK]', '[MASK]')"]
        d1["('Your', '[MASK]', '[MASK]', '[MASK]')"]
        d2["..."]
    end
    subgraph E
        e0["('[MASK]', '[MASK]', '[MASK]', '[MASK]')"]
    end
   
    a0 --> b0 --> c0 --> d0 --> e0
    a1 --> b0
    a2 --> b0
    a3 --> b1 --> c0
    a4 --> b2 --> c1 --> d0
    a5 --> b2
    a6 --> b3 --> c2 --> d1 --> e0

    LeftMarginHack:::phantom ~~~ a0
    e0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

We could train a model to `GuessOneRandomMaskToken` then, at inference, use the learned weight to perform the `GuessNextToken` task instead.
In the scenario where we're data-starved, but we can train for as long as we wish to, with a model as big as it needs to,<d-cite key="ni2025diffusion"></d-cite> and inference is performed using `GuessNextToken`, which model would you put your money on: the one trained solely on this specific `GuessNextToken` task, or the one that saw the worst <d-cite key="kim2025train"></d-cite> that `GuessOneRandomMaskToken` had to offer?


## Shuffling Everything

In the previous section, we saw that both autoregressive language models and mask diffusion language models learn to mimic Alice's message distribution by iteratively `Mash`ing them toward a singleton, reaching a state with no information about anything, thus no information about Alice.
Here we consider the other leading approach to destroying Alice's message: drowning it into irrelevant information.

This mechanism, which I call `Shuffle`, may be more aligned with the historic/physical meaning of the word "diffusion".<d-footnote>In other words, while I expect that some readers could say that `Mash` isn't "real" diffusion, I don't have the same worry for `Shuffle`.</d-footnote>
Whereas `Mash` *may* generate information (e.g., masking order) but will eventually destroy everything to a singleton, `Shuffle` *must* generate new information to be folded into Alice's message, drowning it in the noise.

```mermaid
flowchart LR
    subgraph A
        a0[0]
        a1[1]
        a2[2]
    end
    subgraph B
        b0[0]
        b1[1]
        b2[2]
    end

    a0 --tail---> b0
    a2 --tail---> b0
    a0 --head---> b1
    A ==Shuffle===> B
    a1 --head---> b1
    a2 --head---> b2
    a1 --tail---> b2

    LeftMarginHack:::phantom ~~~ a0
    b0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

If Bob has a good understanding of the `Shuffle` channel, he can model the situation as

$$
\begin{bmatrix}
    \textup{P}(B=0) \cr
    \textup{P}(B=1) \cr
    \textup{P}(B=2)
\end{bmatrix} =
\begin{bmatrix}
    1/2 & 0   & 1/2 \cr
    1/2 & 1/2 & 0   \cr
    0   & 1/2 & 1/2
\end{bmatrix}
\begin{bmatrix}
    \textup{P}(A=0) \cr
    \textup{P}(A=1) \cr
    \textup{P}(A=2)
\end{bmatrix} .
$$

When he observes $B=0$, he knows that Alice said either $0$ or $2$, but he's not sure which: information was destroyed, and he may learn to to generate $A'$, seeking to capture Alice's marginal probability distribution $\textup{P}(A)$.

But notice that (irrelevant) information was also created: when $B=0$, a Bob that understands the channel knows with certainty that the coin flip was "tail" (whereas this coin information was completely unknown to Alice).
In my view, `Shuffle` can destroy information about the original message *because* it generates irrelevant information and has to fold the outcome in as many possible messages than there were before.
The message $A$ can take 3 values, the coin can take two values, the product of the coin and the message can take 6 values, and somehow this has to be crammed into the 3 possible values for $B$.
Something has to give; information about Alice is destroyed.

Next Bob can message Carol through the same `Shuffle` channel, and the same procedure can continue up to Zalgo.

$$
\begin{bmatrix}
    \textup{P}(Z=0) \cr
    \textup{P}(Z=1) \cr
    \textup{P}(Z=2)
\end{bmatrix} =
\begin{bmatrix}
    1/2 & 0   & 1/2 \cr
    1/2 & 1/2 & 0   \cr
    0   & 1/2 & 1/2
\end{bmatrix}^{25}
\begin{bmatrix}
    \textup{P}(A=0) \cr
    \textup{P}(A=1) \cr
    \textup{P}(A=2)
\end{bmatrix}
$$

Now, technically, the probability distribution $\textup{P}(Z)$ still depends on $\textup{P}(A)$, but what does that mean in practice?
Well, it can be shown that

$$
0.33333331 < P(Z=z) < 0.33333335 \quad \forall z \in \{0,1,2\} ,
$$

irrespective of Alice's probability distribution.
The crux of the `Shuffle` learning strategy here amounts to approximating Zalgo as a uniform distribution: 

$$
P(Z'=z') = \frac{1}{3} \quad \forall z' \in \{ 0,1,2 \}.
$$

So, whereas `Mash` eventually takes us to a singleton, `Shuffle` takes us to a known distribution, simple to sample.
Except for this detail, inference for `Shuffle` proceeds the same way as `Mash`: we sample $Z'$ from the known distribution, then apply the respective learned `Guess` down the alphabet until we get $A'$.

Like the previous section adapted the spirit of `Mash` to `MashOneRandomUnmaskedToken`, we can adapt `Shuffle` to obtain `ShuffleOneRandomToken`: a random token position gets substituted by a token selected uniformly at random from the tokenizer's vocabulary.
This is basically the "Uniform" noising process used in SEDD.<d-cite key="lou2024discrete"></d-cite> 

For the rest of this section, let's quickly consider what "shuffling" means in continuous space.
Suppose Alice's message is a $w \times h \times 3$ tensor of real numbers in the $[-1,1]$ interval representing an RGB picture of $w$ pixels wide by $h$ pixels high.
We may define the `GaussianShuffle(δ)` communication channel such that it adds independent Gaussian noise (with mean $0$ and variance $0 < \delta \ll 1$) to each entry of this tensor.

This channel generates (irrelevant) information: for a given message from Alice, there are many possible options for what Bob may receive; arrows branching out means that information is generated.
But the channel also destroys information: there are many messages that Alice could have said that may explain a given message received by Bob; arrows converging in mean that information is destroyed.
As for `Shuffle`, `GaussianShuffle(δ)` destroys information about Alice's message by generating irrelevant information and folding it into limited space.<d-footnote>The story is a bit more complex here because, in a strict mathematical sense, specifying a single real number requires an infinite amount of information. One can work around this by considering differential entropy instead of entropy. However, for computer science applications, there is a much simpler resolution: these "real" numbers are represented as discrete data types. Concretely, a `float32` is really a discrete variable that may take one of $2^{32}$ values, thus capturing at most 32 bits of information.</d-footnote>

The impact of $n$ repeated independent applications of `GaussianShuffle(δ)` is a single "bigger" `GaussianShuffle(n*δ)`.<d-footnote>Recall that the convolution of two Gaussians is a Gaussian whose mean and variance is the sum of their respective means and variances.</d-footnote>
Similarly to how we approximated Zalgo as a uniform distribution earlier, we can choose an $n$ that is high enough so that the outcome of applying `GaussianShuffle(n*δ)` on any image is basically an $w \times h \times 3$ Gaussian with mean zero and variance $n\delta$, i.e., no dependency worth mentioning on the actual image.
In essence, this amounts to the "Variance Exploding" process from Song *et al.* <d-cite key="song2021scorebased"></d-cite>; the "Variance Preserving" version could be similarly obtained by appropriately scaling down the outcome after each addition of Gaussian noise.


## Beyond Markov

In our quest to divide-and-conquer the learning of Alice's distribution, we have seen two different strategies to destroy all the information in her message: `Mash` everything into a singleton, and `Shuffle` long enough to get a maximum-entropy distribution (e.g., uniform or Gaussian).

Both these strategies are Markovian: each person's message has all the information required to get the probability distribution for the next person's message.
Stated differently, the "spine" of all my diagrams looked like this.

```mermaid
flowchart LR
    subgraph A
        a0:::phantom
    end
    subgraph B
        b0:::phantom
    end
    subgraph C
        c0:::phantom
    end
    subgraph D["..."]
        d0:::phantom
    end
    subgraph Z
        z0:::phantom
    end
    A ==> B ==> C ==> D ==> Z
    
    LeftMarginHack:::phantom ~~~ a0
    z0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

But this is not the only possibility, and many diffusion models have a non-Markovian approach to destruction and/or generation.
For example, although DDPM's<d-cite key="ho2020denoising"></d-cite> destroying process may be framed as a Markovian chain, we may also frame it as interpolating between a "clean" sample $A'$ and a "noise distribution" $Z$

$$
B' = \sqrt{\alpha_B} A'  + \sqrt{1 - \alpha_B} Z', \quad C' = \sqrt{\alpha_C} A'  + \sqrt{1 - \alpha_C} Z', \quad \cdots \ ,
$$

given appropriate $1 = \alpha_A \ge \alpha_B \ge \alpha_C \ge \cdots \ge \alpha_Z = 0$.
Below is a diagram showing the Markovian approach on the top-right and the non-Markovian one on the bottom-left.

```mermaid
flowchart LR
    subgraph A
        a0:::phantom
    end
    subgraph B
        b0:::phantom
    end
    subgraph C
        c0:::phantom
    end
    subgraph D["..."]
        d0:::phantom
    end
    subgraph Z
        z0:::phantom
    end
    subgraph Ap["A'"]
        ap0:::phantom
    end
    subgraph Bp["B'"]
        bp0:::phantom
    end
    subgraph Cp["C'"]
        cp0:::phantom
    end
    subgraph Zp["Z'"]
        zp0:::phantom
    end
    subgraph ApxZp["A' × Z'"]
        azp0:::phantom
    end
    A ==> B ==> C ==> D ==> Z
    ApxZp ==> Ap -.- A
    ApxZp ==> Bp -.- B
    ApxZp ==> Cp -.- C
    ApxZp ==> Zp -.- Z
    
    LeftMarginHack:::phantom ~~~ a0
    z0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Here I used $A' \times Z'$ to denote someone that has simultaneous access to both $A'$ and $Z'$ messages, and is thus able to use the previous equation to obtain $B'$, $C'$, *etc.*

But once you've used $A' \times Z'$ to calculate, say, $B'$, you could just *not discard* $Z'$, and obtain $B' \times Z'$.
The same holds up to $Y' \times Z'$ and, with a little more algebra, we can actually go back to $A' \times Z'$ (ignoring finite-precision errors).
I represent such information-preserving back-and-forth conversions using bidirectional arrows.

```mermaid
flowchart LR
    subgraph ApxZp["A' × Z'"]
        apzp0:::phantom
    end
    subgraph BpxZp["B' × Z'"]
        bpzp0:::phantom
    end
    subgraph CpxZp["C' × Z'"]
        cpzp0:::phantom
    end
    subgraph DpxZp["..."]
        dpzp0:::phantom
    end
    subgraph YpxZp["Y' × Z'"]
        ypzp0:::phantom
    end
    subgraph Ap["A'"]
        ap0:::phantom
    end
    subgraph Bp["B'"]
        bp0:::phantom
    end
    subgraph Cp["C'"]
        cp0:::phantom
    end
    subgraph Yp["Y'"]
        yp0:::phantom
    end
    subgraph Zp["Z'"]
        zp0:::phantom
    end
    subgraph BpxZpp["B' × Z''"]
        bpzpp0:::phantom
    end
    subgraph CpxZppp["C' × Z'''"]
        cpzppp0:::phantom
    end
    subgraph YpxZpppp["Y' × Z'''''"]
        ypzpppp0:::phantom
    end
    subgraph App["A''"]
        app0:::phantom
    end
    subgraph A
        a0:::phantom
    end
    subgraph B
        b0:::phantom
    end
    subgraph C
        c0:::phantom
    end
    subgraph D["..."]
        d0:::phantom
    end
    subgraph Y
        y0:::phantom
    end
    subgraph Z
        z0:::phantom
    end
    
    Ap ==> ApxZp <==> BpxZp <==> CpxZp <==> DpxZp <==> YpxZp ==> Zp
    BpxZp ==> Bp ==> BpxZpp -."B'".- BpxZp
    CpxZp ==> Cp ==> CpxZppp -."C'".- CpxZp
    YpxZp ==> Yp ==> YpxZpppp -."Y'".- YpxZp
    A ==> B ==> C ==> D ==> Y ==> Z
    ApxZp ==> App -.- A
    Ap -.- A
    Bp -.- B
    Cp -.- C
    Yp -.- Y
    Zp -.- Z

    LeftMarginHack:::phantom ~~~ ap0
    z0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Projecting $B' \times Z'$ to $B'$ destroys information about $Z'$ and, without it, we cannot get back to $A'$: this projection destroys information about $A'$.
We may thus train a model to use $B'$ to predict the "missing noise" $Z''$, with the intent to obtain a $B' \times Z''$ whose distribution matches $B' \times Z'$.
Equation (14) from DDPM<d-cite key="ho2020denoising"></d-cite> learns such denoising functions.


## About the Diagrams

There exist many types of [probabilistic graphical models](https://en.wikipedia.org/wiki/Graphical_model).
Some are meant to be general-purpose, others are more specialized.
Most are meant for empirical/scientific use: we translate our intuition of what are the non-impossible dependencies between stochastic variables, then we may analyze empirical observations under those assumptions.

This blogpost introduces a new kind of probabilistic graphical model, which I propose to call *generative commutative diagrams*.
Unlike many other kinds of graphical models, those used here are meant for prescription/engineering use.
They read like code: an arrow is like a procedure you can call to transform the origin node into the target node.

As the name "generative commutative diagrams" implies, they have been inspired by the [commutative diagrams](https://en.wikipedia.org/wiki/Commutative_diagram) used by mathematicians, especially [category theorists](https://en.wikipedia.org/wiki/Category_theory).
This blogpost is not about category theory, you do not need to know nor care about category theory to read it, and I am myself definitely *not* an expert on category theory.

Most of the arrows used in this blogpost do *not* represent morphisms, but something less-than-a-morphism.
I call them "harpoons" as I denote them $\rightharpoonup$ when I draw on a whiteboard or in $\texttt{tikz-cd}$, and I reserve $\rightarrow$ for proper morphisms.
For this blogpost, I haven't figured out how to coerce $\texttt{mermaid}$ into making different arrow heads: almost all the arrows show here should be $\rightharpoonup$.
`Project` and `Mash` are among the rare examples that are proper morphisms, which I would usually draw as $\rightarrow$.

An introduction to category theory will often start with something like this.
```mermaid
flowchart LR
    subgraph X
        x0:::phantom
    end
    subgraph Y
        y0:::phantom
    end
    subgraph Z
        z0:::phantom
    end
    X ==f==> Y ==g==> Z
    X ==h==> Z

    LeftMarginHack:::phantom ~~~ x0
    z0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```
They will then mention "this diagram is commutative", which means that $g(f(x)) = h(x)$ for all $x$.
When we show the little arrows between the ~~messages~~ objects, this means that the paths going from $X$ to $Z$ by following $f$ then $g$ must all agree with the paths directly using $h$.

```mermaid
flowchart LR
    subgraph X
        x0["0"]
        x1["1"]
        x2["2"]
    end
    subgraph Y
        y0["0"]
        y1["1"]
    end
    subgraph Z
        z0["0"]
        z1["1"]
    end
    X ==f==> Y ==g==> Z
    X ==h==> Z
    x0 --> y0 --> z0
    x1 --> z0
    x1 --> y0
    x2 --> z1
    x2 --> y1 --> z1
    x0 --> z0

    LeftMarginHack:::phantom ~~~ x0
    z0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

In a generative commutative diagrams, all the big arrows (harpoons and proper morphisms included) **must also satisfy this requirement of commutativity.**

Next, that introduction to category theory may give the following diagram.

```mermaid
flowchart LR
    subgraph X
        x0:::phantom
    end
    subgraph Y
        y0:::phantom
    end
    subgraph Z
        z0:::phantom
    end
    X ==f==> Y ==g==> Z

    LeftMarginHack:::phantom ~~~ x0
    z0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

They will then say that because $f$ and $g$ are morphisms, this picture is equivalent to the first one: we can always combine morphisms head-to-tail to obtain an implicit morphism, here $h = g \circ f$.
**This is not true for my harpoons.**
If either or both $f$ and $g$ are harpoons, the best you can get by chaining them is a way to sample the same probability distribution<d-footnote>Again, these mermaid diagrams use a dashed line to represent "has the same marginal distribution as", but elsewhere I prefer tikz-cd's squiggly line, in reference to the "tilde" symbol that is often used to mean "distributed like".</d-footnote> -- *not* a guarantee that you will end up at the same element.

```mermaid
flowchart LR
    subgraph X
        x0:::phantom
    end
    subgraph Y
        y0:::phantom
    end
    subgraph Z
        z0:::phantom
    end
    subgraph Zp["Z'"]
        zp0:::phantom
    end
    X ==f==> Y ==g==> Z
    X =="g ○ f"==> Zp -.- Z

    LeftMarginHack:::phantom ~~~ x0
    z0 ~~~ RightMarginHack:::phantom
    classDef phantom display: none;
```

Morphisms cannot create information, only destroy or preserve it.
Harpoons can create, preserve and destroy information.

Commutative diagrams that only contain morphisms may only tell "the destruction story", the traditional direction of the data processing inequality.
Harpoons allow you to also tell its dual, "the generation story".

To be clear, we didn't get any real new narrative power: you can tell "the generation story" solely in terms of destruction.<d-footnote>Though you may have to narrate it from the perspective of Laplace's demon, a narrative device that already knows everything there is to be known.</d-footnote>
In that view, harpoons are just "syntactic sugar" for a more involved combination of morphisms.

I have no idea if there is interesting maths to be done with such harpoons and, if yes, perhaps it has already been done under some name that evaded my search.
However, as a specialized kind of graphical model, I've found them particularly useful when thinking about diffusion models and their weird edge cases.

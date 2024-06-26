def step(self, action):
    assert self.lander is not None

    # Update wind and apply to the lander
    assert self.lander is not None, "You forgot to call reset()"
    if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
    ):
        # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
        # which is proven to never be periodic, k = 0.01
        wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
        )
        self.wind_idx += 1
        self.lander.ApplyForceToCenter(
            (wind_mag, 0.0),
            True,
        )

        # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
        # which is proven to never be periodic, k = 0.01
        torque_mag = math.tanh(
            math.sin(0.02 * self.torque_idx)
            + (math.sin(math.pi * 0.01 * self.torque_idx))
        ) * (self.turbulence_power)
        self.torque_idx += 1
        self.lander.ApplyTorque(
            (torque_mag),
            True,
        )

    if self.continuous:
        action = np.clip(action, -1, +1).astype(np.float32)
    else:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid "

    # Apply Engine Impulses

    # Tip is a the (X and Y) components of the rotation of the lander.
    tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

    # Side is the (-Y and X) components of the rotation of the lander.
    side = (-tip[1], tip[0])

    # Generate two random numbers between -1/SCALE and 1/SCALE.
    dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

    m_power = 0.0
    if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
    ):
        # Main engine
        if self.continuous:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            assert m_power >= 0.5 and m_power <= 1.0
        else:
            m_power = 1.0

        # 4 is move a bit downwards, +-2 for randomness
        # The components of the impulse to be applied by the main engine.
        ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
        )
        oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
        )

        impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
        if self.render_mode is not None:
            # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )
            p.ApplyLinearImpulse(
                (
                    ox * MAIN_ENGINE_POWER * m_power,
                    oy * MAIN_ENGINE_POWER * m_power,
                ),
                impulse_pos,
                True,
            )
        self.lander.ApplyLinearImpulse(
            (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
            impulse_pos,
            True,
        )

    s_power = 0.0
    if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
    ):
        # Orientation/Side engines
        if self.continuous:
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            assert s_power >= 0.5 and s_power <= 1.0
        else:
            # action = 1 is left, action = 3 is right
            direction = action - 2
            s_power = 1.0

        # The components of the impulse to be applied by the side engines.
        ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )
        oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )

        # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
        # However, SIDE_ENGINE_HEIGHT is defined as 14
        # This casuses the position of the thurst on the body of the lander to change, depending on the orientation of the lander.
        # This in turn results in an orientation depentant torque being applied to the lander.
        impulse_pos = (
            self.lander.position[0] + ox - tip[0] * 17 / SCALE,
            self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
        )
        if self.render_mode is not None:
            # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (
                    ox * SIDE_ENGINE_POWER * s_power,
                    oy * SIDE_ENGINE_POWER * s_power,
                ),
                impulse_pos,
                True,
            )
        self.lander.ApplyLinearImpulse(
            (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
            impulse_pos,
            True,
        )

    self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    pos = self.lander.position
    vel = self.lander.linearVelocity

    state = [
        (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
        (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
        vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
        vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
        self.lander.angle,
        20.0 * self.lander.angularVelocity / FPS,
        1.0 if self.legs[0].ground_contact else 0.0,
        1.0 if self.legs[1].ground_contact else 0.0,
    ]
    assert len(state) == 8

    terminated = False
    if self.game_over or abs(state[0]) >= 1.0:
        terminated = True
    if not self.lander.awake:
        terminated = True

    reward, individual_reward = self.compute_reward(state, m_power, s_power, terminated)

    if self.render_mode == "human":
        self.render()

    fitness_score = self.compute_fitness_score(state, m_power, s_power, terminated)
    individual_reward.update({'fitness_score': fitness_score})

    return np.array(state, dtype=np.float32), reward, terminated, False, individual_reward

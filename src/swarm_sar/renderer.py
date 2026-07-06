import pygame
from swarm_sar.drone import DroneState


class Renderer:
    def __init__(self, simulator, cell_px: int = 8):
        self.sim = simulator
        self.cell_px = cell_px
        self.show_paths = True
        pygame.init()
        env = simulator.env
        grid_w = env.width * cell_px
        grid_h = env.height * cell_px
        margin = 50
        self.screen = pygame.display.set_mode(
            (grid_w + 2 * margin, grid_h + 2 * margin)
        )
        pygame.display.set_caption("Swarm SAR")
        self.font = pygame.font.SysFont("monospace", 14)
        self.margin = margin
        self.colors = {
            "empty": (200, 200, 200),
            "searched": (80, 200, 80),
            "obstacle": (40, 40, 40),
            "home": (220, 220, 60),
            "drone": (60, 120, 220),
            "path": (160, 200, 255),
            "dead": (220, 60, 60),
        }

    def draw(self):
        env = self.sim.env
        mm = self.sim.mm
        screen = self.screen
        screen.fill((30, 30, 30))
        for y in range(env.height):
            for x in range(env.width):
                rect = pygame.Rect(
                    self.margin + x * self.cell_px,
                    self.margin + y * self.cell_px,
                    self.cell_px,
                    self.cell_px,
                )
                if env.grid[y, x] == 2:
                    pygame.draw.rect(screen, self.colors["home"], rect)
                elif env.grid[y, x] == 1:
                    pygame.draw.rect(screen, self.colors["obstacle"], rect)
                elif (x, y) in mm.searched:
                    pygame.draw.rect(screen, self.colors["searched"], rect)
                else:
                    pygame.draw.rect(screen, self.colors["empty"], rect)
        if self.show_paths:
            for d in self.sim.drones:
                if not d.alive:
                    continue
                for px, py in d.path:
                    rect = pygame.Rect(
                        self.margin + px * self.cell_px,
                        self.margin + py * self.cell_px,
                        self.cell_px,
                        self.cell_px,
                    )
                    pygame.draw.rect(screen, self.colors["path"], rect)
        for d in self.sim.drones:
            cx = (
                self.margin + d.pos[0] * self.cell_px + self.cell_px // 2
            )
            cy = (
                self.margin + d.pos[1] * self.cell_px + self.cell_px // 2
            )
            r = max(self.cell_px // 2 - 1, 2)
            color = self.colors["drone"] if d.alive else self.colors["dead"]
            pygame.draw.circle(screen, color, (cx, cy), r)
            lbl = self.font.render(str(d.id), True, (255, 255, 255))
            screen.blit(lbl, (cx - 4, cy - 6))
        active = sum(1 for d in self.sim.drones if d.alive)
        hud = (
            f"tick:{self.sim.tick} "
            f"coverage:{mm.coverage()*100:.1f}% "
            f"active:{active}/{len(self.sim.drones)}"
        )
        screen.blit(
            self.font.render(hud, True, (255, 255, 255)), (10, 10)
        )
        bx = self.screen.get_width() - 140
        for d in self.sim.drones:
            bh = 10
            fill = int(bh * d.battery / max(self.sim.config.battery_capacity, 1))
            if d.state == DroneState.SEARCHING:
                c = (0, 200, 0)
            elif d.state == DroneState.RETURNING:
                c = (200, 200, 0)
            elif d.state == DroneState.IDLE:
                c = (0, 0, 200)
            else:
                c = (200, 100, 0)
            pygame.draw.rect(
                screen, (50, 50, 50), (bx, 10 + d.id * (bh + 2), 130, bh)
            )
            pygame.draw.rect(
                screen, c, (bx, 10 + d.id * (bh + 2), fill, bh)
            )
        pygame.display.flip()

    def shutdown(self):
        pygame.quit()
